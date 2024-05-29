# CGViT
### Abstract

<div style="text-align: justify;">
The recognition of fruits and vegetables holds significant importance for enhancing processing efficiency, automating harvesting, and facilitating dietary nutrition management. The diverse applications of fruit and vegetable recognition necessitate deployment on end devices with limited resources such as memory and computing power. The key challenge lies in designing lightweight recognition algorithms. However, current lightweight methods still rely on simple CNN-based networks, failing to deeply explore and specifically analyze unique features of fruit and vegetable images to achieve satisfactory recognition performance. To address this challenge, a novel lightweight recognition network termed Channel Grouping Vision Transformer (CGViT) is proposed that utilizes a channel grouping mechanism to capture three discriminative types of features from images and then employs Transformer for feature fusion and global information extraction, ultimately realizing an efficient neural network model for fruit and vegetable recognition. The proposed CGViT approach achieved 71.26%, 99.99%, 98.92%, and 61.33% recognition accuracies on four fruit and vegetable datasets respectively, which outperformed the state-of-the-art methods (MobileViTV2, MixNet, MobileNetV2). The maximum memory usage during training is only 6.48GB, outperforming existing methods. The fruit and vegetable recognition model proposed in this study offers a more profound and effective solution, providing valuable insights for future research and practical applications in this domain.
</div>

### Network structure
![Basic Framework of CGViT](readme_files/structure.png) 

### Images from different datasets
![](readme_files/introduction-2.png) 

### Experimental Results
![](readme_files/table2.png)