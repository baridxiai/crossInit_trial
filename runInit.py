# Name: Barid
#
from transformers import AutoTokenizer
import CrossInit
import logger
import time
import numpy as np
XNLI_LANGS = ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru',  'th', 'tr', 'ur', 'vi', 'zh','sw']
ROOT="~/massive_data/data/wiki/wiki_xlm_gpt"
tokenizer = AutoTokenizer("xlm-mlm-xnli15-1024")
ranking_dict = {}
for lang in XNLI_LANGS:
    with open(ROOT+"/"+lang+"src_256_freq",'r') as file:
        domain_list = dict()
        sorted_domain_list = []
        for k, v in enumerate(file.readlines()):
                c, ids = v.strip().split(" ")
                domain_list[int(ids)] = float(c)
        domain_list = sorted(domain_list.items(), key=lambda x: x[1], reverse=True)
        sorted_domain_list.append([d[0] for d in domain_list])
        ranking_dict[tokenizer.lang2id[lang]] = sorted_domain_list
params = {}
params["emb_dim"] = 1024
params["vocabulary"] = len(tokenizer)
params['ranking'] = ranking_dict
params['bs'] = 32
params['span'] = 1000
params['lr'] = 1e-5
params['freq_num'] = 20000
params['output_path'] = './model_checkpoint'
# params['n_epochs'] =30
params['n_step'] =30*100000
crossInit = CrossInit.CrossInit(params)
logger = logger.create_logger('./train.log')
logger.info('============ Initialized logger ============')
logger.info('\n'.join('%s: %s' % (k, str(v)) for k, v in sorted(dict(vars(params)).items())))
logger.info('The experiment will be stored in %s' % params.output_path)
for n in range(0, params['n_step']):
    loss =crossInit.contrastive_learning_step()
    stats_log = ['%s: %.4f' % (np.mean(loss.cpu().detach().numpy()))]
    stats_log.append('%i samples/s' % int(n_words_proc / (time.time() - tic)))
    logger.info(('%06i - ' % n) + ' - '.join(stats_log))
    # reset
    tic = time.time()
    n_words_proc = 0
    if n %20000:
         crossInit.export()