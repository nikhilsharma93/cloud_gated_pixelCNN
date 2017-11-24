# cloud_gated_pixelCNN
Extension of the Gated-Pixel CNN architecture (https://arxiv.org/abs/1606.05328) to removal of clouds from aerial imagery

For training on the CIFAR-10 dataset, execute 

```
qlua run.lua -b <batch-size>  -r <initial learning rate>  -o <save directory path>  -m <momentum, if used>
```

Requirements: 
`Torch`, `Lua`, `nn, nngraph modules for Torch`

Look under data.lua to change the number of samples used


*Extension to clouds is under construction (via adaptive masking)*

