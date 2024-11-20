from libs_path import *
from libs import *
# torch.cuda.set_device(2)
import time
def main():
    
    args = get_args_1d()
    k = 100
    thred = 1e-2
    beta = 1e-4
    epsilon = 1e-4
    cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    kwargs = {'pin_memory': True} if cuda else {}
    get_seed(args.seed, printout=False)

    data_path = os.path.join(DATA_PATH, 'burgers_data_R10.mat')
    train_dataset = BurgersDataset(subsample=args.subsample,
                                   train_data=True,
                                   train_portion=0.5,
                                   data_path=data_path,)

    valid_dataset = BurgersDataset(subsample=args.subsample,
                                   train_data=False,
                                   valid_portion=100,
                                   data_path=data_path,)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, **kwargs)
    valid_loader = DataLoader(valid_dataset, batch_size=args.val_batch_size, shuffle=False,
                              drop_last=False, **kwargs)

    print("train :", len(train_loader))
    print("valid :", len(valid_loader))
    sample = next(iter(train_loader))
    seq_len = 2**13//args.subsample

    print('='*20, 'Data loader batch', '='*20)
    for key in sample.keys():
        print(key, "\t", sample[key].shape)
    print('='*(40 + len('Data loader batch')+2))

    if is_interactive():
        u0 = sample['node']
        pos = sample['pos']
        u = sample['target']
        _, axes = plt.subplots(nrows=4, ncols=1, figsize=(15, 10))
        axes = axes.reshape(-1)
        indexes = np.random.choice(range(4), size=4, replace=False)
        for i, ix in enumerate(indexes):
            axes[i].plot(pos[ix], u0[ix], label='input')
            axes[i].plot(pos[ix], u[ix, :, 0], label='target')
            axes[i].plot(pos[ix, 1:-1], u[ix, 1:-1, 1],
                         label='target derivative')
            axes[i].legend()

    with open(os.path.join(SRC_ROOT, 'config.yml')) as f:
        config = yaml.full_load(f)
#     test_name = os.path.basename(__file__).split('.')[0]
    config = config["ex1_burgers"]
    config['attn_norm'] = not args.layer_norm
    for arg in vars(args):
        if arg in config.keys():
            config[arg] = getattr(args, arg)
            
    config["seq_len"] = seq_len
    config["k"] = k
    config["attention_type"]="Lrk"
    
    get_seed(args.seed)
    torch.cuda.empty_cache()
    model = SimpleTransformer(**config)
    #model =ResNet(input_width = 1, layer_width = 64) 
    model = model.to(device)
    print(f"\nModel: {model.__name__}\t Number of params: {get_num_params(model)}")

    model_name, result_name = get_model_name(model='Burgers',
                                         num_encoder_layers=config['num_encoder_layers'],
                                         n_hidden=config['n_hidden'],
                                         attention_type=config['attention_type'],
                                         layer_norm=config['layer_norm'],
                                         grid_size=int(2**13//args.subsample),
                                         )
    print(f"Saving model and result in {MODEL_PATH}/{model_name}\n")

#     epochs = args.epochs
    epochs = 100
#     lr = args.lr
    lr = 3e-4
    h = (1/2**13)*args.subsample
    tqdm_mode = 'epoch' if not args.show_batch else 'batch'
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    ### scheduler parameter
    
    max_lr = lr
    div_factor = 1e4
    pct_start = 0.2
    final_div_factor=1e4
    initial_lr = max_lr/div_factor
    min_lr =initial_lr/final_div_factor
    total_steps = len(train_loader)*epochs
    end_step = float(pct_start * total_steps) - 1
    

    loss_func = WeightedL2Loss(regularizer=True, h=h, gamma=args.gamma)

    metric_func = WeightedL2Loss(regularizer=False, h=h)
    
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
            
            
            print("save the model at epoch:", epoch)
            val_result = validate_epoch_darcy(
                model, metric_func, valid_loader, device)
            val_metric = val_result["metric"].sum()
            
            
            
            best_val_epoch = epoch
            best_val_metric = val_metric
            
            best_train_loss = np.inf
            
            torch.save(model.state_dict(), os.path.join(
                model_save_path, model_name))
            best_model_state_dict = {
                        k: v.to('cpu') for k, v in model.state_dict().items()}
            best_model_state_dict = OrderedDict(best_model_state_dict)
            

            
            optimizer = torch.optim.Adam(model.parameters(), lr = lr_history[-1])
            
            L1 = True
            it = 0
            
        if (L1==True) and (best_train_loss>thred) and (( (it+1)%3==0)):
            print("optimize L2 term.")
            L1 = False
            
        
        
        it = it+1
       
        for batch in train_loader:
            loss, _, _ = train_batch_burgers(model, loss_func, batch, optimizer,None, device, grad_clip=grad_clip, 
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
        val_result = validate_epoch_burgers(
                model, metric_func, valid_loader, device)
        loss_val.append(val_result["metric"])
        val_metric = val_result["metric"].sum()
        print("%d: total loss:%.2e, loss1:%.2e, reg:%.2e, ortho:%.2e, norm:%.2e, val_metric:%.2e"%
              (epoch, _loss_mean[0], _loss_mean[1], _loss_mean[2], _loss_mean[3],_loss_mean[4], val_metric)) 
    
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
                k=k_final
                
            )
        save_pickle(result, os.path.join(model_save_path, result_name))
        
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, model_name)))
    model.eval()
    val_metric = validate_epoch_burgers(model, metric_func, valid_loader, device)
    print(f'The final value of rank ：{model.v.shape[0]}')
    print(f"\nBest model's validation metric in this run: {val_metric}")
    
def annealing_cos(start, end, pct):  
    cos_out = math.cos(math.pi * pct) + 1
    return end + (start - end) / 2.0 * cos_out                        
   
   
    

if __name__ == '__main__':
    main()
