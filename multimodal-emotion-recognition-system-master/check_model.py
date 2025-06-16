import torch

# 加载checkpoint
checkpoint = torch.load('best_model.pth', map_location='cpu', weights_only=False)
state_dict = checkpoint['model_state_dict']

# 查找classifier相关的键
classifier_keys = [k for k in state_dict.keys() if 'classifier' in k]
print('Classifier keys:', classifier_keys)

for key in classifier_keys:
    print(f'{key}: {state_dict[key].shape}')

# 查看所有键的前20个
print('\nAll keys (first 20):')
for i, key in enumerate(list(state_dict.keys())[:20]):
    print(f'{i+1}. {key}: {state_dict[key].shape}')

# 查看config
config = checkpoint.get('config', {})
print('\nConfig:', config)