## GitHub放出重大更新：加入免费软件包管理服务

机器之心  *前天*

机器之心报道

**参与：李泽南、杜伟**

> 本周五，代码共享平台 GitHub 发布了一项重要更新：GitHub Package Registry，它可以帮助开发者们轻松查找、管理和发布确保项目正常运行的软件包。

![img](https://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gW96VVlK0wTfxEMfEYibKb8ZBiaGyKsAywy7KyBVib6NwwW0SCicxbQt2iaBSWP2HGQRep5elLQZFhZ0ib1A/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

这也是 GitHub 在 2018 年 6 月[被微软以75亿美元收购](http://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650743216&idx=3&sn=a4cc5c55f7bb96dc5d0825450e6fa887&chksm=871ae5ceb06d6cd8154a6bba8b17ee82dcf9824ceeca1ce64647026a5be6e0d863445031c6be&scene=21#wechat_redirect)以后，公司推出的第一个重要新产品。Package Registry 对于个人用户是免费的，不过 GitHub 表示将会在未来提供付费企业版，并添加围绕安全与合规性的附加功能。

GitHub Package Registry 目前尚处于公开测试阶段，注册使用链接：https://github.com/features/package-registry/signup


![img](https://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gW96VVlK0wTfxEMfEYibKb8ZBWTUZPvpnC0zGYiaBmmpib7IibBibMTAQ7N5SOOWZibTHZ5LITAMbwZyVyQA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

GitHub 的新功能为开发者们解决了大问题，在社交网络上，人们对于 Package Registry 表示出了极高的兴趣。当然，此前承担这种功能的 Maven Central 等产品看来也将因此而退休了。使用 GitHub Package Registry，你可以安全地在自己的开发机构，抑或全世界发布和使用软件包。

![img](https://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gW96VVlK0wTfxEMfEYibKb8ZBRQiaLgHNTTUUCAw49epUJey9ddFfwIibbnpwIxUdoSRK7Qkq4NI6KzkQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)



据 GitHub 产品负责人 Bryan Clark 等人介绍，Package Registry 被设计得非常易于使用，并已支持多种编程语言和工具，如：



- npm (JavaScript)
- Maven (Java)
- RubyGems (Ruby)
- NuGet (.NET)
- Docker images


GitHub 正在努力增加对其他语言的支持，并计划每年推出新功能。

GitHub Package Registry 与 GitHub 完全集成，用户可以使用相同的搜索、浏览和管理工具来查找和发布包，这与存储库的使用方式一样。用户还可以使用相同的用户和团队权限来同时管理代码和包。依托 GitHub 的全球加速 CDN，GitHub Package Registry 可提供快速可靠的下载。



**注册公测版**



**包与代码一起**
在进行一个依赖于包的项目时，用户通常需要信任这些包，了解它们的代码，并且需要与创建它们的团队联系。在组织内部，你需要快速找出被允许使用的内容。GitHub Package Registry 使用户能够更方便地使用同一个 GitHub 界面来查找 GitHub 上的任意公共包或者自己组织、存储库内部的私有包。

![img](https://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gW96VVlK0wTfxEMfEYibKb8ZBFUWvzaibmwu984kDk2zYJqcKxXwasjticahHyx0PhzMWeAnCpXxkS7gQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

GitHub Package Registry 与常见包管理客户端兼容，用户可自行选择工具来发布包。如果用户的存储库更复杂，则能够发布不同类型的多个包。同时，借助于 webhooks 和 GitHub Actions，用户可以完全自定义自己的发布和发布后工作流程。

发布一个开源包？大多数开源项目在 GitHub 上有自己的代码，用户可以先发布包的预览版本（prerelease version），在社区内部进行测试，之后就可以将包的特定版本（specific version）推介到自己选择的 public registry。



**统一的标识和权限**



![img](https://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gW96VVlK0wTfxEMfEYibKb8ZBPEeZJXUGgV4E1dfK3nIuwNHoaZ9prBcia98Xtxrlf75IYDHu0giasZyw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)*GitHub 的个人页面新增了一个名为「Packages」的选项卡，其中会列出帐户或机构拥有的软件包*



如果用户在管理代码和包时使用不同的系统，则必须保留不同的用户凭证和权限。现在，用户可以使用兼容二者的单一凭证，并使用相同工具管理访问权限。GitHub 上的包延续了与存储库关联的可见性和权限，组织也无需跨系统维护单独的 package registry 和镜像权限（mirror permissions）。



**包视图**


![img](https://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gW96VVlK0wTfxEMfEYibKb8ZBBCm6REbSpQhUGt1avLD3uc6hXzFiaNMpE64uibdXwl5HVL9DGKSU6CGg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)



托管在 GitHub 上的包通常包括 details、下载统计以及完整的历史记录，用户可以清楚地看到。因此，用户很容易就能找到并使用适合自己项目的包。如果用户发布的包有更详细的视图，则可以准确了解其他人和存储库如何使用它们。



**加入公测**



GitHub Package Registry 目前处于公开测试阶段，GitHub 欢迎程序员们加入使用的行列，并推动其发展。GitHub 似乎已经成为微软一直在追寻的大规模社交网络平台，有开发者表示，自己已在使用 GitHub 与潜在的雇主进行交流。当然，在这里与其他开发者进行互动，并结交新的朋友也是很常见的事。*![img](https://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gW8Zfpicd40EribGuaFicDBCRH6IOu1Rnc4T3W3J1wE0j6kQ6GorRSgicib0fmNrj3yzlokup2jia9Z0YVeA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)*



*参考链接：*

*https://github.com/features/package-registryhttps://github.blog/2019-05-10-introducing-github-package-registry/https://www.businessinsider.com/github-package-registry-open-source-tools-2019-5*