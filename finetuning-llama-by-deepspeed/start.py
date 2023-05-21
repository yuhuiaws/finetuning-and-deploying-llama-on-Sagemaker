import os
import json
import argparse
import socket

if __name__ == "__main__":
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--P', type=string)
#     parser.add_argument('--c', type=string)
#     args, _ = parser.parse_known_args()
    
    hosts = json.loads(os.environ['SM_HOSTS'])
    current_host = os.environ['SM_CURRENT_HOST']
    host_rank = int(hosts.index(current_host))
    
    master = json.loads(os.environ['SM_TRAINING_ENV'])['master_hostname']
    master_addr = socket.gethostbyname(master)
    
    #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
    os.environ['NODE_INDEX'] = str(host_rank)
    os.environ['SM_MASTER'] = str(master)
    os.environ['SM_MASTER_ADDR'] = str(master_addr)
    os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
    
    os.system("chmod +x ./2-example.sh")
    os.system("/bin/bash -c ./2-example.sh")