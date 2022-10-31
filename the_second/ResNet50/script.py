import paddle
from paddle.regularizer import L2Decay
import paddlex as pdx
from paddlex import transforms as T


# 定义训练和验证时的transforms
train_transforms = T.Compose([
    T.RandomCrop(
        crop_size=224,
        scaling=[.88, 1.],
        aspect_ratio=[3. / 4, 4. / 3]),
    T.RandomHorizontalFlip(prob=0.0),
    T.RandomVerticalFlip(prob=0.0), T.RandomDistort(
        brightness_range=0.9,
        brightness_prob=0.5,
        contrast_range=0.9,
        contrast_prob=0.5,
        saturation_range=0.9,
        saturation_prob=0.0,
        hue_range=18.0,
        hue_prob=0.0), T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
eval_transforms = T.Compose([
    T.ResizeByShort(short_size=256),
    T.CenterCrop(crop_size=224), T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = pdx.datasets.ImageNet(
    data_dir=r'PATH', # 训练集
    file_list=r'PATH', # 训练集list文档
    label_list=r'PATH', # 训练集label
    transforms=train_transforms,
    shuffle=True)

eval_dataset = pdx.datasets.ImageNet(
    data_dir=r'PATH', # 测试集
    file_list=r'PATH', # 测试集list
    label_list=r'PATH', # 测试集label
    transforms=eval_transforms)


num_classes = len(train_dataset.labels)
model = pdx.cls.PPLCNet_ssld(num_classes=num_classes)

# 定义优化器：使用CosineAnnealingDecay
train_batch_size = 16
step_each_epoch = train_dataset.num_samples // train_batch_size
learning_rate = paddle.optimizer.lr.CosineAnnealingDecay(
            learning_rate=.001, T_max=step_each_epoch * 100)
optimizer = paddle.optimizer.Momentum(
        learning_rate=learning_rate,
        momentum=.9,
        weight_decay=L2Decay(1e-04),
        parameters=model.net.parameters())

model.train(
        num_epochs=100,
        train_dataset=train_dataset,
        train_batch_size=train_batch_size,
        eval_dataset=eval_dataset,
        save_interval_epochs=1,
        log_interval_steps=2,
        save_dir=r'PATH', # 模型存储文档
        pretrain_weights=r'PATH',
        optimizer=optimizer,
        use_vdl=True,
        resume_checkpoint=None)
