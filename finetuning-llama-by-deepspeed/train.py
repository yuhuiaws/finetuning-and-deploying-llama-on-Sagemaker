from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
from transformers.models.llama.tokenization_llama import LlamaTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_from_disk
import random
import logging
import sys
import argparse
import os
import torch
import subprocess
import deepspeed
import torch.distributed as dist

if __name__ == "__main__":

    # Environment variables set by torch.distributed.launch
    LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    WORLD_RANK = int(os.environ['RANK'])
    
    dist.init_process_group(backend='nccl', rank=WORLD_RANK, world_size=WORLD_SIZE)
    
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    #parser.add_argument("--eval_steps",type=int,default=5000)
    parser.add_argument("--learning_rate", type=str, default=2e-5)
    parser.add_argument("--evaluation_strategy",type=str,default="epoch")
    parser.add_argument("--gradient_accumulation_steps",type=int,default=4)
    parser.add_argument("--c",type=bool,default=False)
    #parser.add_argument("--logging_steps",type=int,default=5000)
    parser.add_argument("--save_steps",type=int,default=500)
    parser.add_argument("--save_strategy",type=str,default="steps")
    parser.add_argument("--save_total_limit",type=int,default=4)
    parser.add_argument("--model_max_length",type=int,default=512)
    parser.add_argument("--bf16",type=bool,default=True)

    # Data, model, and output directories
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    
    parser = deepspeed.add_config_arguments(parser)

    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # load datasets
    train_dataset = load_from_disk(args.training_dir)
    test_dataset = load_from_disk(args.test_dir)

    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f" loaded test_dataset length is: {len(test_dataset)}")

    # compute metrics function for binary classification
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    model_name_or_path = "/tmp/llama_source/"
    #Download source model from S3 using s5cmd for local rank 0
    if LOCAL_RANK == 0:
        print("-----------local rank 0 downloading model from s3----")
        os.system("./s5cmd sync {0} {1}".format(os.environ['SOURCE_MODEL_BEFORE_TRAINING_S3_PATH'], model_name_or_path))
    
    #Note: the barrier is used to ensure just only local rank 0 to download model assets from s3. 
    torch.distributed.barrier()
        
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,use_cache=False)
    
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, model_max_length=args.model_max_length,padding_side="right")
    
    num_new_tokens = tokenizer.add_special_tokens({'additional_special_tokens': ['[STOP]','[SEP]']})
    if tokenizer.pad_token is None:
        num_new_tokens += tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    print("We have added", num_new_tokens, "tokens")
    '''
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
    '''
    
    # define training args
    training_args = TrainingArguments(
        output_dir="/tmp/intermediate",
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        warmup_steps=args.warmup_steps,
        #warmup_ratio = 0.03,
        #lr_scheduler_type = "cosine",
        #max_grad_norm = 0.7,
        evaluation_strategy="no",         #just for test
        logging_dir=f"{args.output_data_dir}/logs",
        logging_steps = 10,
        gradient_checkpointing=True,
        learning_rate=float(args.learning_rate),
        deepspeed=args.deepspeed_config,
        #save_steps = args.save_steps,
        save_strategy = "no",          #just for test
        save_total_limit = args.save_total_limit,
        save_on_each_node = True,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        fp16=True,  
        bf16=False,  # Use BF16 if available
    )
  
    # create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        #compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer
    )

    # train model
    trainer.train()

    #We now save the model assets to an intermediate path.
    #Note: plesae do not save the model into /opt/ml/model (because Sagemaker will tar and compress all of files under /opt/ml/model, and it will consume much time for LLM.)
    print("------saving model!-----")
    save_model_dir = '/tmp/output/asset/'
    tokenizer.save_pretrained(save_model_dir)
    trainer.save_model(save_model_dir)
    print("------model is saved!-----")
    
    #Note: we just use the rank 0 process to upload the trained model assets to S3 by s5cmd command.
    if WORLD_RANK == 0:
        os.system("./s5cmd sync {0} {1}".format(save_model_dir, os.environ['TARGET_MODEL_AFTER_TRAINING_S3_PATH']))
    
    #Note: we should sync with every ranker and ensure only global rank 0 uploading the model assets successfully. 
    torch.distributed.barrier()
