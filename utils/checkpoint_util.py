""" Util functions for loading and saving checkpoints """
import os
import torch

def load_trained_checkpoint(model, checkpoint_path, name):
    # load trained model
    model_dict = model.state_dict()
    if checkpoint_path is not None:
        print('Load old model from trained checkpoint...')
        trained_dict = torch.load(os.path.join(checkpoint_path, name))['params']
        pretrained_dict = {k: v for k, v in trained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        raise ValueError('Trained checkpoint must be given.')
    return model

def save_train_checkpoint(model, output_path, name='checkpoint.tar'):
    torch.save(dict(params=model.state_dict()), os.path.join(output_path, name + '_checkpoint.tar'))

def save_classifer_checkpoint(classifer, output_path, name='checkpoint.tar'):
    torch.save(dict(params=classifer.state_dict()), os.path.join(output_path, name + '_classifer_checkpoint.tar'))

def load_encoder_checkpoint(model, encoder_checkpoint_path, name):
    # load pretrained model for point cloud encoding
    model_dict = model.state_dict()
    if encoder_checkpoint_path is not None:
        print('Load encoder module from checkpoint...')
        pretrained_dict = torch.load(os.path.join(encoder_checkpoint_path, name))['params']
        pretrained_dict = {'encoder.' + k: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        raise ValueError('Encoder checkpoint must be given.')
    return model