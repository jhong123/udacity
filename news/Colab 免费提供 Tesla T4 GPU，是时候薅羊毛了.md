## Colab 免费提供 Tesla T4 GPU，是时候薅羊毛了

机器之心  *1周前*

机器之心报道

**机器之心编辑部**

> 近日，Colab 全面将 K80 替换为 Tesla T4，新一代图灵架构、16GB 显存，免费 GPU 也能这么强。



想要获取免费算力？可能最常见的方法就是薅谷歌的羊毛，不论是 Colab 和 Kaggle Kernel，它们都提供免费的 K80 GPU 算力。不过虽然 K80 这种古董级的 GPU 也能提供可观的算力，但我们发现用于试验模型越来越不够用了。尤其最近的 Transformer 或 GPT-2 等复杂模型，不是训练迭代时间长，就是被警告显存已满。



最近，Colab 在 Twitter 官方账户上表示，现在已经可以免费用 T4 GPU 了，它不仅能够提供更多的计算力，同时还提供更大的显存：



![img](https://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gWib3FZaWBBz3j3LiaRs9wsGdrriayfwJ55MgEDrB36jw7X74N97cuSuJPsmWbpmN81po6fjF86cfVsuQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)



看到这条信息，小编也是挺激动的，终于有了更强大的免费算力，我们马上在 Colab 上查看 GPU 的使用情况。如下我们看到 Colab 现在确实使用的是 Tesla T4 GPU，而且显存也达到了 16 GB，比以前 K80 12GB 的显存又要大了一圈。



![img](https://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gWib3FZaWBBz3j3LiaRs9wsGdrUmsb7uiceTohlxZ8E8OdMTupyrnHKJsiacl18EIhaicVGrlgQItjJoEEg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)



**图灵架构下的 Tesla T4**



T4 GPU 适用于许多[机器学习](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw%3D%3D&mid=2650761098&idx=1&sn=d9526bf18e24ca5b97428bb13430dc2e)、可视化和其它 GPU 加速工作负载。每个 T4 GPU 具有 16GB 的内存，它还提供最广泛的精度支持（FP32、FP16、INT8 和 INT4），以及英伟达 Tensor Core 和 RTX 实时可视化技术，能够执行高达 260 TOPS 的计算性能。



Tesla T4 采用全新的图灵（Turing）架构，相比过去的架构 Pascal，它在 Shader Compute 的基础上增加了具备 AI 训练和推理能力的 Tensor Core 和支持光线跟踪的 RT Core。



**机器学习推理能力**



在众多 GPU 中，T4 是运行推理工作的很好选择，尽管我们在 Colab 中大多都用于训练。T4 在 FP16、INT8 和 INT4 的高性能特性让你能实现灵活的准确率/性能权衡，并运行大规模模型推理过程，而这些在其它 GPU 上很难做到的。T4 的 16GB 显存支持大型机器学习模型，在[图像生成](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw%3D%3D&mid=2650761098&idx=1&sn=d9526bf18e24ca5b97428bb13430dc2e)或[机器翻译](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw%3D%3D&mid=2650761098&idx=1&sn=d9526bf18e24ca5b97428bb13430dc2e)等耗显存的任务中，Colab 能运行地更流畅了。



谷歌计算引擎上的机器学习推理性能高达 4267 张图像/秒，而延迟低至 1.1 毫秒。但考虑到 T4 的价格、性能、全球可用性和高速的谷歌网络，在计算引擎上用 T4 GPU 运行产品工作负载也是一个很好的解决方案。



**机器学习训练能力**



V100 GPU 凭借其高性能计算、Tensor Core 技术和 16GB 大显存，能支持较大的机器学习模型，已成为在云端训练机器学习模型的主要 GPU。而 T4 以更低的成本支持所有这些，这使得它成为扩展分布式训练或低功率试验的绝佳选择。T4 拥有 2560 个 CUDA 核心，对于我们在 Colab 试验模型已经足够了。



T4 GPU 可以很好地补充 V100 GPU，它虽然没有那么 V100 剽悍，但相比 K80 已经有很多进步了。而且由于 T4 非常节能，替换掉 K80 在能耗上也能降低不少。



如下展示了 T4 和 V100 之间的差别，T4 支持多精度加速，确实非常适合做推理，以后将预训练模型放在 Colab 上也是极好的。



![img](https://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gWib3FZaWBBz3j3LiaRs9wsGdr4Lsn9xBGmyzgLOibvLn8kWNsKWUW0iakUkzgKdo6EKLHADWdLMbcFBDg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

*T4 与 V100 之间的算力对比，其中 T4 在谷歌云每小时大概需要 0.95 美元，不过目前已经面向 Colab 免费提供了。*



**K80 与 T4 到底有什么不同**



2014 年发布的 K80 采用的是 Kepler 架构，而 2018 年发布的 T4 采用的是 Turing 架构，从时间上来说中间还差着 Volta、Pascal、Maxwell 三大架构。



K80 主要具有以下特性：



- 带有双 GPU 设计的 4992 个 NVIDIA CUDA 内核，可显著加速应用程序性能
- 通过 NVIDIA GPU 加速提升双精度浮点性能至 2.91 Teraflops
- 通过 NVIDIA GPU 加速提升单精度浮点性能至 8.73 Teraflops



T4 提供革命性的多精度推理性能，以加速现代人工智能的各种应用。T4 封装在节能的小型 70 瓦 PCIe 中，而 K80 当时的能耗达到了 300W，所以 T4 的效率高了很多。



T4 的性能规格如下：



![img](https://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gWib3FZaWBBz3j3LiaRs9wsGdrRhEIAYgaibKMCK5HmNa87dQq8s6Q90BWPGxY36mKxIQrYicwic50eBkLw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)



**一路走来的 Colaboratory**



现在，快来试试 Colab 吧，这种免费算力不用岂不可惜？其实自从一年多前谷歌发布 Colab，它就已经吸引了非常多研究者与开发者的目光。刚开始虽然提供免费算力，但并不能称得上好用，我们总会感觉有一些「反人类」的设计，例如 cd 命令不太能 work、文件管理系统不健全、难以与谷歌云端硬盘交互、不能使用 TensorBoard 等等。



但造成这些的原因其实是我们不太了解 Colab 的特性。很多时候甚至不看文档与教程，感觉和 Jupyter Notebook 一样简单，因此就直接上手了。但是随着该项目的不断发展，很多问题都解决了，很多新特性都完善了，至少现在我们用起来真的很爽。



现在我们看看 Colab 近来的新特性吧～



2018 年 10，Colab 加了一个文档浏览器。这非常便利，我们只需点击一下就可以上传和下载文件。



![img](https://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gWib3FZaWBBz3j3LiaRs9wsGdrKsWjeNVW90LlbqiaR45JurKzib0ib2aGfUYZibDkfOGC2ERpah3icuiaUapg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)



同年 10 月份，Colab 免费提供 TPU 算力了，它提供 8 个核心的免费算力，即 4 块 TPU 芯片。



![img](https://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gWib3FZaWBBz3j3LiaRs9wsGdrBmeuo8dfiaB1d9C4YRrD8ibVKYYftIxyb6TOhp5eHZzhoWDSxISfrPdQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)



到了今年 2 月份，Colab 又提供了一种全新的暗黑系主题，这也是我们现在常用的。



![img](https://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gWib3FZaWBBz3j3LiaRs9wsGdricze3VnB6ebhByuG6ftjGVO5OZTiachPWAevlAoZUJbWEKTwYrfy7iaAw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)



与此同时，Colab 也开始支持 TensorBoard，只需要使用魔术命令符「%」就能可视化训练过程。



![img](https://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gWib3FZaWBBz3j3LiaRs9wsGdru3QMrmYAHQoaqqUYH5GUUQO63npfJ0OxV1yaY7gfGiaGd5fIicBNcXmw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)



当然这里只列举了少量更新，还有很多修正与改进没有提到。例如与 GitHub 私有库连接、提供交互式 TF 教程、以及文本图像预览等等。再加上现在提供 Tesla T4，Colab 也许会越来越好用，在上面开源的实现、项目、教程也会越来越多。*![img](https://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gW8Zfpicd40EribGuaFicDBCRH6IOu1Rnc4T3W3J1wE0j6kQ6GorRSgicib0fmNrj3yzlokup2jia9Z0YVeA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)*