# myASR
实时ASR(whisper系列、funasr系列) + 翻译（Helsinki、M2M100） + AI总结(暂未完成)

1. 使用说明

（1）下载对应的ASR、翻译模型 放到models文件夹

（2）通过终端，进入到src目录，执行以下python命令 

    python main.py funasr paraformer-speech_68m m2m100 zh-en

    python main.py whisper small m2m100 en-zh

（3）参数说明：

① 使用的ASR模型系列（Whisper/funasr） 
② ASR模型名称(small(whisper)、paraformer-speech_68m(funasr)) 
③ 翻译模型(m2m100/helsinki)  
④ 翻译语种对（源语言-目标语言） eg: zh-en

2. windows一键包

https://drive.google.com/file/d/1N_j7x-Sa8gCPG1tfyJW0UDHBrBzyyHU7/view?usp=drive_link

下载打开后，运行 funasr.bat 或者 whisper.bat 