use tch::{Tensor, nn, nn::ModuleT};

#[derive(Debug)]
pub struct ConvNN {
    conv1: nn::Conv2D,  //第一个卷积层用于提取低级别的特征
    conv2: nn::Conv2D,  //第二个卷积层则用于提取更高级别的特征
    fc1: nn::Linear,    //第一个线性层用于将卷积层提取的特征组合起来
    fc2: nn::Linear,    //第二个线性层则用于输出最终的预测结果
}

impl ConvNN {
    pub fn new(vs: &nn::Path) -> ConvNN {
        let conv1 = nn::conv2d(vs, 1, 8, 65, Default::default());
        let conv2 = nn::conv2d(vs, 8, 16, 65, Default::default());
        let fc1 = nn::linear(vs, 102400, 1024, Default::default());
        let fc2 = nn::linear(vs, 1024, 5, Default::default());
        ConvNN { conv1, conv2, fc1, fc2 }
    }
}

impl ModuleT for ConvNN {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {   //神经网络的前向传播步骤
        xs.view([-1, 1, 512, 512])                        //将输入张量xs重塑为形状为[-1, 1, 512, 512]的张量。这里的-1表示该维度的大小由其他维度自动推断，1表示输入通道数，512表示图像的高度和宽度。
            .apply(&self.conv1)                          //将第一个卷积层应用于输入张量。
            .max_pool2d_default(2)                //使用池化层来进行最大池化。这样可以将每个卷积层输出的空间大小减半。
            .apply(&self.conv2)                          //将第二个卷积层应用于池化后的张量。
            .max_pool2d_default(2)                //再次对卷积层的输出进行最大池化，池化核大小为2。
            .view([-1, 102400])                         //将池化后的张量重塑为形状为[-1, 102400]的张量，以便输入到全连接层。
            .apply(&self.fc1)                            //将第一个线性层应用于重塑后的张量。
            .relu()                                      //对线性层的输出应用ReLU激活函数。
            .dropout(0.5, train)                      //对激活后的张量应用dropout，丢弃概率为0.5。如果train参数为真，则应用dropout，否则不应用。
            .apply(&self.fc2)                                  //将第二个线性层应用于dropout后的张量，得到最终的输出。
    }
}