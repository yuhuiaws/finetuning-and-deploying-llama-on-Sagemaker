# finetuning-and-deploying-llama-on-Sagemaker

Use the two different methods (deepspeed and SageMaker model parallelism/SMP library) to fine tune llama model on Sagemaker. Then deploy the fine tuned llama on Sagemaker with server side batch. 

## There is four parts in this repo:
1. Prepare dataset: You can use the prepare-data-for-llama.ipynb and open source dataset such as dialy-dialogue.txt.zip to prepare dataset for llama.
2. Deploy fine tuned llama on SageMkaer: We use Large Model Inference/LMI container to deploy llama on SageMaker. Also, the demo code can perform the server side batch in order to improve the throughput. (The code is suitable for the case which is single sample/prompt per client request)
3. Fine tune llama by deepspeed on SageMaker multiple nodes: We use deepspeed and torch.distributed.launch to fine tune llama. 
4. Fine tune llama by SMP on SageMaker multiple nodes: We use SMP+HF trainer API to fine tune the llama zero code change.


