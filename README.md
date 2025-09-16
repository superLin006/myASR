# myASR
实时ASR(whisper系列、funasr系列) + 多语种翻译（Helsinki、M2M100） + AI总结(可切换任意LLM)

1. 使用说明

（1）创建虚拟环境
```bash
    conda create  -n myAsr python=3.8  -y 
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y # GPU版本Pytorch
    pip install -r requirements.txt  
```
（2）下载对应的ASR、翻译模型 放到models文件夹（需手动取消python代码注释一个个加载）
```bash
      python download_models.py
```
（2）通过终端启动，进入到src目录，执行以下python命令 
```bash
    python main.py funasr paraformer-speech_68m m2m100 zh-en
    python main.py whisper small m2m100 en-zh
```
（3）参数说明：

① 使用的ASR模型系列（Whisper/funasr） 

② ASR模型名称(  small(whisper)、paraformer-speech_68m(funasr)  ) 

③ 翻译模型(m2m100/helsinki)  

④ 翻译语种对（源语言-目标语言） eg: zh-en

2. windows一键包

https://drive.google.com/file/d/1N_j7x-Sa8gCPG1tfyJW0UDHBrBzyyHU7/view?usp=drive_link

下载打开后，运行 funasr.bat 或者 whisper.bat 