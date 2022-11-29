import sys

def call_model(args, device):
    from ad_model import RNN_enc_RNN_clf, Transformer_enc_RNN_clf, Transformer_enc_DNN_clf
    if args.encoder == 'rnn' and args.classifier == 'rnn':
        model = RNN_enc_RNN_clf(args)
    elif args.encoder == 'transformer' and args.classifier == 'rnn':
        model = Transformer_enc_RNN_clf(args)
    elif args.encoder == "transformer" and args.classifier == "dnn":
        model = Transformer_enc_DNN_clf(args)
    else:
        print("encoder and classifier mismatch")
        sys.exit(-1)

    model = model.to(device)

    return model

def get_const(data_name):
    # split to nodes
    if data_name == 'wsd':
        n_nodes = 5
        n_features = 23
    elif data_name in ["lad1", "lad2"]:
        n_nodes = 4
        n_features = 23
    elif 'KDD' in data_name:
        n_nodes = 1
        n_features = 121
    elif 'unsw' in data_name:
        n_nodes = 1
        n_features = 196
    elif 'cicids2017' in data_name:
        n_nodes = 1
        n_features = 78
    elif 'int' in data_name:
        n_nodes = 1
        n_features = 7
    else:
        print('[ad_utils.get_const]data_name must be wsd, lad1, or lad2')
        import sys; sys.exit(-1)
    return n_nodes, n_features
