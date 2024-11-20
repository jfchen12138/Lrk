from libs_path import *
from libs import *
import time
from timeit import default_timer
import gc
torch.cuda.set_device(4)

def main():
    args = get_args_2d()
    k = 500
    cuda = not args.no_cuda and torch.cuda.is_available()
    if cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if cuda else 'cpu')
    kwargs = {'pin_memory': True} if cuda else {}
    get_seed(args.seed)

    train_path = os.path.join(DATA_PATH, 'piececonst_r421_N1024_smooth1.mat')
    test_path = os.path.join(DATA_PATH, 'piececonst_r421_N1024_smooth2.mat')
    train_dataset = DarcyDataset(data_path=train_path,
                                 subsample_attn=args.subsample_attn,
                                 subsample_nodes=args.subsample_nodes,
                                 train_data=True,
                                 train_len = 1024,)

    valid_dataset = DarcyDataset(data_path=test_path,
                                 normalizer_x=train_dataset.normalizer_x,
                                 subsample_attn=args.subsample_attn,
                                 subsample_nodes=args.subsample_nodes,
                                 train_data=False,
                                 valid_len=100)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, **kwargs)
    valid_loader = DataLoader(valid_dataset, batch_size=args.val_batch_size, shuffle=False,
                              drop_last=False, **kwargs)

    n_grid = int(((421 - 1)/args.subsample_nodes) + 1)
    n_grid_c = int(((421 - 1)/args.subsample_attn) + 1)
    print(n_grid,n_grid_c)
    downsample, upsample = DarcyDataset.get_scaler_sizes(n_grid, n_grid_c,
                                                         scale_factor=not args.no_scale_factor)
    print("downsample:",downsample)
    print("upsample:",upsample)
    sample = next(iter(train_loader))

    print('='*20, 'Data loader batch', '='*20)
    for key in sample.keys():
        print(key, "\t", sample[key].shape)

    print('='*(40 + len('Data loader batch')+2))


    with open(os.path.join(SRC_ROOT, 'config.yml')) as f:
        config = yaml.full_load(f)
    # test_name = os.path.basename(__file__).split('.')[0]
    config = config["ex2_darcy"]
    config['normalizer'] = train_dataset.normalizer_y.to(device)
    config['downscaler_size'] = downsample
    config['upscaler_size'] = upsample
    config['attn_norm'] = not args.layer_norm
    if config['attention_type'] == 'fourier' or n_grid < 211:
        config['norm_eps'] = 1e-7
    elif config['attention_type'] == 'galerkin' and n_grid >= 211:
        config['norm_eps'] = 1e-5

    for arg in vars(args):
        if arg in config.keys():
            config[arg] = getattr(args, arg)

    config["seq_len"] = sample['pos'].shape[1]
    config["k"] = k

    torch.manual_seed(seed=args.seed)
    torch.cuda.manual_seed(seed=args.seed)

    torch.cuda.empty_cache()
    model = FourierTransformer2D(**config)

    model = model.cuda()
    epochs = args.epochs
#     epochs = 160
#     if config['attention_type'] in ['fourier', 'softmax']:
#         lr = min(args.lr, 5e-4)
#     else:
#         lr = args.lr
    lr = args.lr
    h = 1/n_grid
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = OneCycleLR(optimizer, max_lr=lr, 
                           div_factor=1e4, 
                           final_div_factor=1e4,
                           pct_start=0.3,
                           steps_per_epoch=len(train_loader), 
                           epochs=epochs)

    loss_func = WeightedL2Loss2d(regularizer=True, h=h, gamma=args.gamma)

    metric_func = WeightedL2Loss2d(regularizer=False, h=h)

    model_name, result_name = get_model_name(model='darcy_lrk%d_adapt'%k,
                                                 num_encoder_layers=config['num_encoder_layers'],
                                                 n_hidden=config['n_hidden'],
                                                 attention_type=config['attention_type'],
                                                 layer_norm=config['layer_norm'],
                                                 grid_size=n_grid,
                                                 inverse_problem=False,
                                                 additional_str=f'32f'
                                                 )
    
    
    loss_epoch = []
    loss_train = []
    loss_val = []
    best_val_metric = np.inf
    best_val_epoch = None
    epsilon = 1e-3
    epoch =1
#     best_sparse = np.inf
    while epoch < epochs+1:

        model.train()
        print("epoch:", epoch)
        t1 = default_timer()
        for batch in train_loader:
            loss, _, _= train_batch_darcy(model, loss_func, batch, optimizer, scheduler, device, grad_clip= 0.999)
            loss = np.array(loss)
            loss_epoch.append(loss)
            _loss_mean = np.mean(loss_epoch, axis=0)
        print("train:%.4f %.4f %.4f %.4f"%(_loss_mean[0],_loss_mean[1],_loss_mean[2], _loss_mean[3]) ) 
       
       
        loss_train.append(_loss_mean)
        loss_epoch = []
        val_result = validate_epoch_darcy(model, metric_func, valid_loader, device)
        loss_val.append(val_result["metric"]) 
        val_metric = val_result["metric"]
        print("valid : %.4f"% val_metric)
        t2 = default_timer()
        print("time:", t2 - t1)
        print("\n")
        
     
        if val_metric <= best_val_metric:
            param_dict = model.state_dict()
            best_val_metric = val_metric
            best_val_epoch = epoch
            torch.save(model.state_dict(), os.path.join(
                                    MODEL_PATH, model_name)) 
                             
        
        
        epoch=epoch+1
    print(f"best model epoch {best_val_epoch}:{best_val_metric}")
    v = param_dict["v"]
    v_true = v**2
    print("temp:",v)
    print("v:",torch.sort(v_true, descending=True))
    param_dict["v"] = v[v_true>epsilon]
    param_dict["s"] = param_dict["s"][:, v_true>epsilon]
    param_dict["d"] = param_dict["d"][v_true>epsilon, :]
    print("the value of k :", k)
    k = len(param_dict["v"])
    
    config["k"] = k
    print("the value of k after updating:", k)
    
    
    del model
    model = model = FourierTransformer2D(**config)
    model = model.cuda()
    
    model.load_state_dict(param_dict)
    model.eval()
    val_metric = validate_epoch_darcy(model, metric_func, valid_loader, device)
    print(f"\nBest model's validation metric in this run: {val_metric}")
    
    
    
    
    np.save("/home/jfchen/lin-and-lrk/adaptive_result/darcy_adptive_lrk%d_train"%k,loss_train)
    np.save("/home/jfchen/lin-and-lrk/adaptive_result/darcy_adptive_lrk%d_valid"%k,loss_val)
    
if __name__ == "__main__":
    main()