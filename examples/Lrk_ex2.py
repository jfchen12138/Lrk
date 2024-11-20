from libs_path import *
from libs import *
import time
torch.cuda.set_device(1)
def main():
 
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

    with open(os.path.join(SRC_ROOT, 'config.yml')) as f:
        config = yaml.full_load(f)
#     test_name = os.path.basename(__file__).split('.')[0]
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
    config["attention_type"]="Lrk"
    
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
    tqdm_mode = 'epoch' if not args.show_batch else 'batch'
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = WeightedL2Loss2d(regularizer=True, h=h, gamma=args.gamma)

    metric_func = WeightedL2Loss2d(regularizer=False, h=h)


    ### scheduler parameter
    
    max_lr = lr
    div_factor = 1e4
    pct_start = 0.3
    final_div_factor=1e4
    initial_lr = max_lr/div_factor
    min_lr =initial_lr/final_div_factor
    total_steps = len(train_loader)*epochs
    end_step = float(pct_start * total_steps) - 1
    

   
    
    loss_epoch = []
    loss_train = []
    loss_val = []
    lr_history = []
    grad_clip = 0.999
    model_save_path=MODEL_PATH

    
    best_train_loss = np.inf
    best_val_metric = np.inf
    L1 = False
    

    k_final = k
    
    step_num = 0
    it=0
    
    for epoch in range(epochs):

        
        if (epoch%13==0) and (best_train_loss< thred):
            
            v_temp = model.v**2
            v = model.v[v_temp > epsilon]
            s = model.s[:, v_temp>epsilon]
            d = model.d[v_temp>epsilon, :]
            model.v = nn.Parameter(v)
            model.s = nn.Parameter(s)
            model.d = nn.Parameter(d)
            print(f'The number of v after discarding：{model.v.shape[0]}')
            k_final = model.v.shape[0]
            
            val_result = validate_epoch_darcy(
                model, metric_func, valid_loader, device)
            val_metric = val_result["metric"].sum()
            
            best_train_loss = np.inf
            
            best_val_epoch = epoch
            best_val_metric = val_metric
            torch.save(model.state_dict(), os.path.join(
                model_save_path, model_name))
            best_model_state_dict = {
                        k: v.to('cpu') for k, v in model.state_dict().items()}
            best_model_state_dict = OrderedDict(best_model_state_dict)
            print("save model at epoch:", epoch)
            

            
            optimizer = torch.optim.Adam(model.parameters(), lr = lr_history[-1])
            
            L1 = True
            it=0
        
        
        if (L1==True) and (best_train_loss>thred) and (( (it+1)%3==0)):
            print("optimize L2 term.")
            L1 = False
       
        it = it+1
        
        for batch in train_loader:
            loss, _, _ = train_batch_darcy(model, loss_func, batch, optimizer,None, device, grad_clip=grad_clip, 
                                                 L1=L1, beta = beta)
            step_num = step_num + 1
            
            if step_num <= end_step:
                pct =step_num /end_step
                lr = annealing_cos(initial_lr, max_lr, pct)
                optimizer.param_groups[0]["lr"] = lr
                                 
            else:
                pct = (step_num - end_step)/(total_steps - 1 - end_step)
                lr = annealing_cos(max_lr, min_lr, pct)
                optimizer.param_groups[0]["lr"] = lr
                                 
            
            loss = np.array(loss)
            loss_epoch.append(loss)
            
            lr = optimizer.param_groups[0]['lr']
            lr_history.append(lr)
            _loss_mean = np.mean(loss_epoch, axis=0)
           
        loss_train.append(_loss_mean)
        if _loss_mean[1] < best_train_loss:
            best_train_loss = _loss_mean[1]
        loss_epoch = []
        
        val_result = validate_epoch_darcy(
                model, metric_func, valid_loader, device)
        loss_val.append(val_result["metric"])
        
        val_metric = val_result["metric"].sum()
        print("%d: total loss:%.2e, loss1:%.2e, reg:%.2e, norm:%.2e, val_metric:%.2e"%
              (epoch, _loss_mean[0], _loss_mean[1], _loss_mean[2],_loss_mean[3], val_metric)) 
    
        if val_metric < best_val_metric:
            best_val_epoch = epoch
            best_val_metric = val_metric
            torch.save(model.state_dict(), os.path.join(
                model_save_path, model_name))
            best_model_state_dict = {
                        k: v.to('cpu') for k, v in model.state_dict().items()}
            best_model_state_dict = OrderedDict(best_model_state_dict)
            print("save the model at epoch:", epoch)
        
        result = dict(
                best_val_epoch=best_val_epoch,
                best_val_metric=best_val_metric,
                loss_train=np.asarray(loss_train),
                loss_val=np.asarray(loss_val),
                lr_history=np.asarray(lr_history),
                best_model=best_model_state_dict,
                optimizer_state=optimizer.state_dict(),
                k = k_final
            )
        save_pickle(result, os.path.join(model_save_path, result_name))
        
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, model_name)))
    model.eval()
    val_metric = validate_epoch_darcy(model, metric_func, valid_loader, device)
    print(f'The final value of rank ：{model.v.shape[0]}')
    print(f"\nBest model's validation metric in this run: {val_metric}")
    
def annealing_cos(start, end, pct):  
    cos_out = math.cos(math.pi * pct) + 1
    return end + (start - end) / 2.0 * cos_out                        
   
   
    

if __name__ == '__main__':
    main()

    