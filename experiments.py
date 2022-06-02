experiments = [
    {
        'exp_number': 1,
        'adapter_tasks': ['mnli', 'qqp', 'sst', 'wgrande', 'boolq'],
        'train_tasks': ['imdb', 'mrpc', 'argument', 'scitail'],
        'test_tasks': ['sick', 'rte', 'cb']
    },
    {
        'exp_number': 3,
        'adapter_tasks': ['hswag', 'siqa', 'cqa', 'csqa', 'wgrande'],
        'train_tasks': ['imdb', 'mrpc', 'argument', 'scitail', 'mnli'],
        'test_tasks': ['sick', 'rte', 'cb', 'qqp', 'sst', 'boolq']
    },
    {
        'exp_number': 4,
        'adapter_tasks': ['cb', 'mnli', 'rte', 'scitail', 'sick'],
        'train_tasks': ['argument', 'imdb', 'mrpc', 'qqp'],
        'test_tasks': ['wgrande', 'boolq', 'sst']
    },
    {
        'exp_number': 5,
        'adapter_tasks': ['boolq','cb','mrpc','imdb','csqa'],
        'train_tasks': ['mnli','qqp','sst','wgrande'],
        'test_tasks': ['sick','rte','argument','scitail']
    },
    {
        'exp_number': 6,
        'adapter_tasks': ['mnli','qqp','sst','wgrande','imdb','scitail','argument','boolq','mrpc','sick','rte','cb', 'hswag', 'siqa', 'cqa', 'csqa'],
        'train_tasks': ['imdb', 'mrpc', 'argument', 'scitail'],
        'test_tasks': ['sick','rte','cb']
    },
    {
        'exp_number': 7,
        'adapter_tasks': ['mnli','qqp','sst','wgrande','imdb','scitail','argument','boolq','mrpc','sick','rte','cb', 'hswag', 'siqa', 'cqa', 'csqa'],
        'train_tasks': ['imdb', 'mrpc', 'argument', 'scitail', 'mnli'],
        'test_tasks': ['sick','rte','cb', 'qqp', 'sst', 'boolq']
    },
    {
        'exp_number': 8,
        'adapter_tasks': ['mnli','qqp','sst','wgrande','imdb','scitail','argument','boolq','mrpc','sick','rte','cb', 'hswag', 'siqa', 'cqa', 'csqa'],
        'train_tasks': ['imdb', 'mrpc', 'argument', 'qqp'],
        'test_tasks': ['wgrande', 'boolq', 'sst']
    },
    {
        'exp_number': 9,
        'adapter_tasks': ['mnli','qqp','sst','wgrande','imdb','scitail','argument','boolq','mrpc','sick','rte','cb', 'hswag', 'siqa', 'cqa', 'csqa'],
        'train_tasks': ['mnli','qqp','sst','wgrande'],
        'test_tasks': ['sick','rte','argument', 'scitail']
    },
    {
        'exp_number': 10, # exp A
        'adapter_tasks': ['mnli','scitail'],
        'train_tasks': ['sick', 'rte'],
        'test_tasks': ['cb']
    },
    {
        'exp_number': 11, # exp B
        'adapter_tasks': ['mnli','sick'],
        'train_tasks': ['rte','cb'],
        'test_tasks': ['scitail']
    },
    {
        'exp_number': 12, # exp C
        'adapter_tasks': ['rte','cb'],
        'train_tasks': ['mnli','scitail'],
        'test_tasks': ['sick']
    },
    {
        'exp_number': 13, # exp D
        'adapter_tasks': ['wgrande', 'hswag', 'siqa', 'cqa', 'csqa'],
        'train_tasks': ['argument','sst','scitail','qqp', 'sick'],
        'test_tasks': ['boolq','cb','rte', 'mrpc', 'imdb', 'mnli']
    },
    {
        'exp_number': 14, # exp E
        'adapter_tasks': ['mnli','qqp','sst','wgrande','imdb','scitail','argument','boolq','mrpc','sick','rte','cb', 'hswag', 'siqa', 'cqa', 'csqa'],
        'train_tasks': ['argument','sst','scitail','qqp', 'sick'],
        'test_tasks': ['boolq','cb','rte', 'mrpc', 'imdb', 'mnli']
    }
]