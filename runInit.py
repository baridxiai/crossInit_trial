# Name: Barid
#
import wandb
from transformers import AutoTokenizer
from CrossInit import CrossInit
import time
import logger
import numpy as np
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
#os.environ['PYTORCH_CUDA_ALLOC_CONF']="expandable_segments:True,max_split_size_mb:24"
ALL_LANG=["af","am","ar","as","az","be","bg","bn","br","bs","ca","cs","cy","da","de","el","en","eo","es","et","eu","fa","fi","fr","fy","ga","gd","gl","gu","ha","he","hi","hr","hu","hy","id","is","it","ja","jv","ka","kk","km","kn","ko","ku","ky","la","lo","lt","lv","mg","mk","ml","mn","mr","ms","my","ne","nl","no","om","or","pa","pl","ps","pt","ro","ru","sa","sd","si","sk","sl","so","sq","sr","su","sv","sw","ta","te","th","tl","tr","ug","uk","ur","uz","vi","xh","yi","zh"]
#XNLI_LANGS = ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru',  'th', 'tr', 'ur', 'vi', 'zh','sw']
XNLI_LANGS = [ 'de', 'en',  'hi']
# ROOT=os.path.expanduser("~") + "/massive_data/data/wiki/wiki_xlm_gpt"
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("xlm-mlm-xnli15-1024")
ranking_dict = {}
def get_ranking_dict():
    for j,lang in enumerate(XNLI_LANGS):
        with open("./word_freq/"+lang+"freq.txt",'r') as file:
            domain_list = []
            for k, v in enumerate(file.readlines()):
                if k>100000:
                    break
                c = v.strip().split(" ")
                if c[0] != "INFO:" and  c[0] != "Processing":
                    domain_list.append(c[0])
            ranking_dict[j] = tokenizer(domain_list,add_special_tokens=False)
    return ranking_dict
params = {}
params["emb_dim"] = 1024
params["vocabulary"] = len(tokenizer)
params['ranking'] = get_ranking_dict()
params['bs'] = 256
params['span'] = 1500
params['lr'] = 1e-4
params['freq_num'] = 20000
params['output_path'] = './crossInit_embedding'
params["name"] = "XLM3-CrossInit-with-span-" + str(params['span'])
params["tokenizer"] = tokenizer
wandb.init(
    # Set the project where this run will be logged
    project="cross-init",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": params['lr'],
    },
)
# params['n_epochs'] =30
params['n_step'] =30*100000
crossInit = CrossInit(params)
logger = logger.create_logger('./train.log')
logger.info('============ Initialized logger ============')
# logger.info('\n'.join('%s: %s' % (k, str(v)) for k, v in sorted(dict(vars(params)).items())))
logger.info('The experiment will be stored in %s' % params['output_path'])
tic = time.time()
if __name__ == "__main__":
    for n in range(0, params['n_step']):
        loss =crossInit.contrastive_learning_step()
        # reset
        wandb.log({"loss": np.mean(loss.cpu().detach().numpy())})
        if n %1000==999:
            crossInit.export()