<p align="center">
  <h1 align="center"><img src="asset/logo.png" width="256"></h1>
  <h1 align="center">MMDU: A Multi-Turn Multi-Image Dialog Understanding Benchmark and Instruction-Tuning Dataset for LVLMs</h1>
    <p align="center">
    <a href="https://github.com/Liuziyu77"><strong>Ziyu Liu</strong></a>
    ·
    <a href="https://github.com/SunzeY"><strong>Tao Chu</strong></a>
    ·
    <a href="https://yuhangzang.github.io/"><strong>Yuhang Zang</strong></a>
    ·
    <strong>Xilin Wei</strong>
    ·
    <a href="https://lightdxy.github.io/"><strong>Xiaoyi Dong</strong></a>
    ·
    <a href="https://panzhang0212.github.io/"><strong>Pan Zhang</strong></a>
    ·
    <strong>Zijian Liang</strong>
    ·
     <a href="http://yjxiong.me/"><strong>Yuanjun Xiong</strong></a>
      ·
     <strong>Yu Qiao</strong>
    ·
     <a href="http://dahua.site/"><strong>Dahua Lin</strong></a>
    ·
     <a href="https://myownskyw7.github.io/"><strong>Jiaqi Wang</strong></a>
  </p>
  <h2 align="center">Submitted to arXiv</h2>
  📖<a href="https://arxiv.org/pdf/2403.13805.pdf">Paper</a> |🏠<a href="https://liuziyu77.github.io/MMDU/">Homepage</a></h3>|🤗<a href="https://huggingface.co/datasets/laolao77/MMDU">Huggingface</a></h3>
<div align="center"></div>
<p align="center">
  <p>
In this paper, we highlight the potential of combining <strong>retrieving and ranking</strong> with multi-modal large language models to revolutionize perception tasks such as fine-grained recognition, zero-shot image recognition, and few-shot object recognition. Motivated by the limited zero-shot/few-shot of CLIP and MLLMs on fine-grained datasets, our <strong>RAR</strong> designs the pipeline that uses MLLM to rank the retrieved results. Our proposed approach can be seamlessly integrated into various MLLMs for real-world applications where the variety and volume of categories continuously expand. Our method opens up new avenues for research in augmenting the MLLM’s abilities with the retrieving-augmented solution and could be beneficial for other tasks such as reasoning and generation in future works.
  </p>
  <a href="">
    <img src="asset/pipeline.png" alt="Logo" width="100%">
  </a>
<br>

## 📢 News
- 🚀 [06/11/2024] We upload our MMDU-45k dataset to huggingface.
- 🚀 [06/11/2024] We upload our MMDU benchmark to huggingface.
- 🚀 [06/11/2024] Our work is submitted to arXiv.

## 💡 Highlights
- 🔥 **Multi-turn and Multi-image**: Our benchmark showcases a conversational setting with a maximum of 20 images and 17 turns, thereby surpassing the scope of preceding works and authentically replicating real-world chat assistant interactions. 
- 🔥 **Long Context**: With a maximum of 18k text+image tokens, MMDU evaluates the capacity of LVLMs to process and comprehend extended contextual information with a long context history.
- 🔥 **Open-ended Evaluation**: Departing from traditional benchmarks that rely on close-ended questions with concise outputs (eg, multiple-choice questions or short answers), our benchmark adopts a more realistic and nuanced approach, assessing LVLM's performance through free-form multi-turn outputs that prioritize scalability and explainability.

## 💎 MMDU Benchmark
Although many LVLMs now claim to handle tens of thousands, hundreds of thousands, or even millions of tokens in length, their actual performance significantly declines in real-world applications as the number of images or the length of the context increases. Both the dialogue quality and image recognition capabilities of LVLMs deteriorate notably under these conditions.

To evaluate the multi-image multi-turn dialogue capabilities of existing models, we have developed the MMDU Benchmark. Our benchmark comprises **110 high-quality multi-image multi-turn dialogues with more than 1600 questions**, each accompanied by detailed long-form answers. Previous benchmarks typically involved only single images or a small number of images, with fewer rounds of questions and short-form answers. However, MMDU significantly increases the number of images, the number of question-and-answer rounds, and the in-context length of the Q&A. The questions in MMUD **involve 2 to 20 images**, with **an average image&text token length of 8.2k tokens**, and **a maximum image&text length reaching 18K tokens**, presenting significant challenges to existing multimodal large models. 

<a href="">
  <img src="asset/statistic.png" alt="Logo">
</a>

## 🎆 MMDU-45k Instruct Tuning Dataset
In the MMDU-45k, we construct a total of 45k instruct tuning data conversations. Each data in our MMDU-45k dataset features an ultra-long context, with **an average image&text token length of 5k** and **a maximum image&text token length of 17k** tokens. Each dialogue contains **an average of 9 turns of Q&A**, with **a maximum of 27 turns**. Additionally, each data includes content from 2-5 images. The dataset is constructed in a well-designed format, providing excellent scalability. It can be expanded to generate a larger number and longer multi-image, multi-turn dialogues through combinations. ***The image-text length and the number of turns in MMDU-45k significantly surpass those of all existing instruct tuning datasets.*** This enhancement greatly improves the model's capabilities in multi-image recognition and understanding, as well as its ability to handle long-context dialogues.

## 🎇 Finetune with MMDU-45k
The model fine-tuned with MMDU-45k has shown significant improvements in multi-image recognition and long-text dialogue capabilities. As demonstrated in the following case, the fine-tuned InternLM-Xcomposer2 is able to provide richer responses and more accurate visual information compared to before.
<a href="">
  <img src="asset/case.png" alt="case">
</a>
Additionally, the model fine-tuned with MMDU-45k has shown performance improvements on eight benchmarks, including MMBench, MMvet, and MMMU.

## 🛠️ Usage
We will release the evaluation code soon!!!

## ✒️Citation
```
@misc{liu2024rar,
      title={RAR: Retrieving And Ranking Augmented MLLMs for Visual Recognition}, 
      author={Ziyu Liu and Zeyi Sun and Yuhang Zang and Wei Li and Pan Zhang and Xiaoyi Dong and Yuanjun Xiong and Dahua Lin and Jiaqi Wang},
      year={2024},
      eprint={2403.13805},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## 📄 License
![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg) ![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg) **Usage and License Notices**: The data and code are intended and licensed for research use only.
