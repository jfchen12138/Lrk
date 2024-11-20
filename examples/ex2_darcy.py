from libs_path import *
from libs import *
import time
torch.cuda.set_device(1)
def main():
    ###
#     k = 100
#     thred = 1e-2
#     alpha = 0.3
#     beta = 0.01
#     epsilon = 1e-4
#     lr=3e-4

###   pow
#     k = 100
#     thred = 1e-2
#     alpha = 0.3
#     beta = 0.05
#     epsilon = 1e-4
    
    
    
    args = get_args_2d()
    k = 100
    thred = 1e-2
    beta = 1
    epsilon = 1e-4
    cuda = not args.no_cuda and torch.cuda.is_available()
    if cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if cuda else 'cpu')
    kwargs = {'pin_memory': True} if cuda else {}
    get_seed(args.seed)

    args.batch_size = 2
    args.val_batch_size=2

    train_path = os.path.join(DATA_PATH, 'piececonst_r421_N1024_smooth1.mat')
    test_path = os.path.join(DATA_PATH, 'piececonst_r421_N1024_smooth2.mat')
    train_dataset = DarcyDataset(data_path=train_path,
                                 subsample_attn=args.subsample_attn,
                                 subsample_nodes=args.subsample_nodes,
                                 train_data=True,
                                 train_len=1024,)

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

    if is_interactive():
        idx = 3
        a = sample['node']
        u = sample['target']
        elem = train_dataset.elem
        node = train_dataset.pos
        ah = F.interpolate(a[..., 0].unsqueeze(1), size=(
            n_grid_c, n_grid_c), mode='bilinear', align_corners=True)
        uh = train_dataset.normalizer_x.inverse_transform(u)
        uh = F.interpolate(uh[..., 0].unsqueeze(1), size=(
            n_grid_c, n_grid_c), mode='bilinear', align_corners=True)
        ah = ah[idx].numpy().reshape(-1)
        uh = uh[idx].numpy().reshape(-1)
        showsolution(node, elem, uh, width=600, height=500)

        fig, ax = plt.subplots(figsize=(10, 10))
        ah_plot = ax.imshow(ah.reshape(n_grid_c, n_grid_c), cmap='RdBu')
        fig.colorbar(ah_plot, ax=ax, anchor=(0, 0.3), shrink=0.8)
        

    with open(os.path.join(SRC_ROOT, 'config.yml')) as f:
        config = yaml.full_load(f)
    test_name = os.path.basename(__file__).split('.')[0]
    config = config[test_name]
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
    
  
    model = model.to(device)
    
    print(
        f"\nModel: {model.__name__}\t Number of params: {get_num_params(model)}")
    model_name, result_name = get_model_name(model='darcy',
                                             num_encoder_layers=config['num_encoder_layers'],
                                             n_hidden=config['n_hidden'],
                                             attention_type=config['attention_type'],
                                             layer_norm=config['layer_norm'],
                                             grid_size=n_grid,
                                             inverse_problem=False,
                                             additional_str=f'32f'
                                             )
    print(f"Saving model and result in {MODEL_PATH}/{model_name}\n")

    epochs = args.epochs
    lr = 2e-4
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
    
    L1 = False
    
    start = time.time()
    
    
    result = run_train(model, loss_func, metric_func,
                       train_loader, valid_loader,
                       optimizer, scheduler,
                       train_batch=train_batch_darcy,
                       validate_epoch=validate_epoch_darcy,
                       epochs=epochs,
                       patience=None,
                       tqdm_mode='epoch',
                       model_name=model_name,
                       result_name=result_name,
                       device= device,
                       L1 = L1,
                       beta = beta)
    
    
    end = time.time()
    mean_time = (end-start)/epochs
    
    
    path = os.path.join(SRC_ROOT, 'high_hidden_result')
    train_time_path = os.path.join(path, "darcy_%s_training_time_per_epoch.npy"%args.attention_type)
    np.save(train_time_path, mean_time)
    
    if args.attention_type == "Lrk":
        param = torch.load(os.path.join(MODEL_PATH, model_name))
        v_temp = param["v"]
#         v = v_temp**2
        v = torch.abs(v_temp)
        print("v", torch.sort(v, descending=True))
        param["v"] = param["v"][v>epsilon]
        param["s"] = param["s"][:, v>epsilon]
        param["d"] = param["d"][v>epsilon, :]
        k = len(param["v"])
        config["k"] = k
        print("the value of k after updating:", k)
        
        
        del model
        model = FourierTransformer2D(**config)
        model = model.cuda()
        model.load_state_dict(param)
        torch.save(model, os.path.join(MODEL_PATH, model_name))
        
        model.eval()
        val_metric = validate_epoch_darcy(model, metric_func, valid_loader, device)
        print(f" model's validation metric after pruning in this run: {val_metric}")
    
    else:
    
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH, model_name)))
        torch.save(model,os.path.join(MODEL_PATH, model_name))
        model.eval()
        val_metric = validate_epoch_darcy(model, metric_func, valid_loader, device)
        print(f"\nBest model's validation metric in this run: {val_metric}")
    
    
    
#     training_curve_path = os.path.join(path, "darcy_%s_training_loss.npy"%args.attention_type)
#     valid_curve_path = os.path.join(path, "darcy_%s_valid_loss.npy"%args.attention_type)
    
    
    train_loss = result["loss_train"]
    val_loss = result["loss_val"]
#     np.save(training_curve_path, train_loss)
#     np.save(valid_curve_path, val_loss)
    
    if args.attention_type == "fourier":
        sample = next(iter(valid_loader))
        a = sample['node']
        pos = sample['pos']
        u = sample['target']
        grid = sample['grid']
        with torch.no_grad():
            model.eval()
            out_ = model(a.to(device), None, pos.to(device), grid.to(device))
            out = out_['attn_weights']


        eigenvalue_path = os.path.join(path, 'darcy attention')
        torch.save(out, eigenvalue_path) 

if __name__ == '__main__':
    main()
