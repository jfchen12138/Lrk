"""
(2+1)D Navier-Stokes equation + Galerkin Transformer
MIT license: Paper2394 authors, NeurIPS 2021 submission.
"""
from libs_path import *
from libs import *
from libs.ns_lite import *
import time
torch.cuda.set_device(3)
get_seed(1127802)

attn_type = "galerkin"
k = 100
thred = 1e-2
# alpha = 0.3
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
scheduler = OneCycleLR(optimizer, max_lr=lr, div_factor=1e4, final_div_factor=1e4,
                       steps_per_epoch=len(train_loader), epochs=epochs)

loss_func = WeightedL2Loss2d(regularizer=True, h=h, gamma=0.1)

metric_func = WeightedL2Loss2d(regularizer=False, h=h)

start = time.time()
result = run_train(model, loss_func, metric_func,
                   train_loader, valid_loader,
                   optimizer, scheduler,
                   train_batch=train_batch_ns,
                   validate_epoch=validate_epoch_ns,
                   epochs=epochs,
                   patience=None,
                   tqdm_mode='batch',
                   model_name=model_name,
                   result_name=result_name,
                   mode='min',
                   device=device,
                   L1 = False,
                   beta = beta)

end = time.time()
mean_time = (end-start)/epochs
    

path = os.path.join(SRC_ROOT, 'high_hidden_result')
train_time_path = os.path.join(path, "ns_%s_training_time_per_epoch.npy"%config["attention_type"])
np.save(train_time_path, mean_time)

if config["attention_type"] == "Lrk":
    param = torch.load(os.path.join(MODEL_PATH, model_name))
    model.load_state_dict(param)
    model.eval()
    val_metric = validate_epoch_ns(model, metric_func, valid_loader, device)
    print(f"\nBest model's validation metric in this run: {val_metric}")
    v_temp = param["v"]
    v = v_temp**2
    print("v", torch.sort(v, descending=True))
    param["v"] = param["v"][v>epsilon]
    param["s"] = param["s"][:, v>epsilon]
    param["d"] = param["d"][v>epsilon, :]
    k = len(param["v"])
    config["k"] = k
    print("the value of k after updating:", k)


    del model
    model = FourierTransformer2DLite(**config)
    model = model.cuda()
    model.load_state_dict(param)
    torch.save(model, os.path.join(MODEL_PATH, model_name))

    model.eval()
    val_metric = validate_epoch_ns(model, metric_func, valid_loader, device)
    print(f" model's validation metric after pruning in this run: {val_metric}")

else:

    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, model_name)))
    torch.save(model,os.path.join(MODEL_PATH, model_name))
    model.eval()
    val_metric = validate_epoch_ns(model, metric_func, valid_loader, device)
    print(f"\nBest model's validation metric in this run: {val_metric}")



# training_curve_path = os.path.join(path, "ns_%s_training_loss.npy"%config["attention_type"])
# valid_curve_path = os.path.join(path, "ns_%s_valid_loss.npy"%config["attention_type"])


train_loss = result["loss_train"]
val_loss = result["loss_val"]
# np.save(training_curve_path, train_loss)
# np.save(valid_curve_path, val_loss)

if config["attention_type"] == "fourier":
    sample = next(iter(valid_loader))
    a = sample['node']
    pos = sample['pos']
    u = sample['target']
    grid = sample['grid']
    with torch.no_grad():
        model.eval()
        out_ = model(a.to(device), None, pos.to(device), grid.to(device))
        out = out_['attn_weights']


    eigenvalue_path = os.path.join(path, 'ns attention')
    torch.save(out, eigenvalue_path) 



"""
4 GT layers: 48 d_model
2 SC layers: 20 d_model for spectral conv with 12 Fourier modes
Total params: 862049

diag 0 + xavier 1e-2, encoder dp = ffn dp = 5e-2
    3.406e-03 at epoch 99

diag 1e-2 + xavier 1e-2, encoder dp 0, ffn dp = 5e-2
    3.078e-03 at epoch 100
"""
