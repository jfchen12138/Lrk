from libs_path import *
from libs import *
from libs.ns_lite import *
import time
torch.cuda.set_device(2)
get_seed(1127802)

def main():
    attn_type = "Lrk"
    k = 100
    thred = 1e-2
#     alpha = 0.3
    beta = 0.01
    epsilon = 1e-4

    data_path = os.path.join(DATA_PATH, 'ns_V1e-3_N5000_T50.mat')
    train_dataset = NavierStokesDatasetLite(data_path=data_path,
                                            train_data=True,)
    valid_dataset = NavierStokesDatasetLite(data_path=data_path,
                                            train_data=False,)
    batch_size = 2
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              drop_last=False,
                              pin_memory=True)

    sample = next(iter(train_loader))

    print('='*20, 'Data loader batch', '='*20)
    for key in sample.keys():
        print(key, "\t", sample[key].shape)

    print('='*(40 + len('Data loader batch')+2))

    n_grid = 64

    config = defaultdict(lambda: None,
                         node_feats=10+2,
                         pos_dim=2,
                         n_targets=1,
                         n_hidden=196,  # attention's d_model
                         num_feat_layers=0,
                         num_encoder_layers=4,
                         n_head=4,
                         dim_feedforward=392,
                         attention_type= attn_type,
                         feat_extract_type=None,
                         xavier_init=0.01,
                         diagonal_weight=0.01,
                         layer_norm=True,
                         attn_norm=False,
                         return_attn_weight=True,
                         return_latent=False,
                         decoder_type='ifft',
                         freq_dim=20,  # hidden dim in the frequency domain
                         num_regressor_layers=2,  # number of spectral layers
                         fourier_modes=12,  # number of Fourier modes
                         spacial_dim=2,
                         spacial_fc=False,
                         dropout=0.0,
                         encoder_dropout=0.0,
                         decoder_dropout=0.0,
                         ffn_dropout=0.05,
                         debug=False,
                         )


    config["seq_len"] = sample['pos'].shape[1]
    config["k"] = k


    torch.cuda.empty_cache()
    model = FourierTransformer2DLite(**config)
    # print(get_num_params(model))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(
            f"\nModel: {model.__name__}\t Number of params: {get_num_params(model)}")

    model_name, result_name = get_model_name(model='ns',
                                                 num_encoder_layers=config['num_encoder_layers'],
                                                 n_hidden=config['n_hidden'],
                                                 attention_type=config['attention_type'],
                                                 layer_norm=config['layer_norm'],
                                                 grid_size=n_grid,
                                                 inverse_problem=False)

    print(f"Saving model and result in {MODEL_PATH}/{model_name}\n")



    epochs = 100
    lr = 5e-4
    h = 1/64
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = OneCycleLR(optimizer, max_lr=lr, div_factor=1e4, final_div_factor=1e4,
    #                        steps_per_epoch=len(train_loader), epochs=epochs)

    loss_func = WeightedL2Loss2d(regularizer=True, h=h, gamma=0.1)

    metric_func = WeightedL2Loss2d(regularizer=False, h=h)

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
    it = 0

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
            val_result = validate_epoch_ns(
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


            k_final = model.v.shape[0]



            optimizer = torch.optim.Adam(model.parameters(), lr = lr_history[-1])

            L1 = True
            
            it=0
            
        if (L1==True) and (best_train_loss>thred) and (( (it+1)%3==0)):
            print("optimize L2 term.")
            L1 = False
       
        it = it+1


        for batch in train_loader:
            loss, _, _ = train_batch_ns(model, loss_func, batch, optimizer,None, device, grad_clip=grad_clip, 
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
        val_result = validate_epoch_ns(
                model, metric_func, valid_loader, device)
        loss_val.append(val_result["metric"])
        val_metric = val_result["metric"].sum()
        print("%d: total loss:%.2e, loss1:%.2e, error:%.2e, reg:%.2e,norm:%.2e, val_metric:%.2e"%
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
                k = k_final

            )
        save_pickle(result, os.path.join(model_save_path, result_name))

    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, model_name)))
    model.eval()
    val_metric = validate_epoch_ns(model, metric_func, valid_loader, device)
    print(f'The final value of rank ：{model.v.shape[0]}')
    print(f"\nBest model's validation metric in this run: {val_metric}")

def annealing_cos(start, end, pct):  
    cos_out = math.cos(math.pi * pct) + 1
    return end + (start - end) / 2.0 * cos_out                        
   
   
    

if __name__ == '__main__':
    main()


