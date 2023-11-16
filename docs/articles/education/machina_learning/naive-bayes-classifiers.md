# Naive Bayes Classifiers
> # 朴素贝叶斯分类器

Hey friends! 👋 It's me, Miss Neura, here today to unpack the Naive Bayes classifier.
> 嘿，朋友们！👋 是我，Neura小姐，今天在这里解开朴素贝叶斯分类器。

Now I know "naive" doesn't sound very flattering in the name. 😅 But don't let that fool you!
> 现在我知道“天真”这个名字听起来不是很讨人喜欢。😅 但不要让它欺骗你！

Naive Bayes is actually a super simple yet powerful algorithm for classification tasks like spam detection and sentiment analysis. 🎉
> 朴素贝叶斯实际上是一种超级简单但功能强大的算法，用于垃圾邮件检测和情感分析等分类任务。🎉

It works by calculating conditional probabilities based on Bayes' theorem and an assumption of independence between features.
> 它的工作原理是根据贝叶斯定理和特征之间独立性的假设来计算条件概率。

I know that sounds a little math-y, but stick with me! 🤓 I'll break down Bayes and the "naive" assumption piece by piece in easy-to-understand terms.
> 我知道这听起来有点数学，但请坚持下去！🤓 我将用通俗易懂的术语逐一分解贝叶斯和“幼稚”假设。

By the end, you'll have a clear understanding of how Naive Bayes ingests data to make predictions for categorical variables. 📈 The key is maximizing those probabilities!
> 最后，您将清楚地了解朴素贝叶斯如何摄取数据以对分类变量进行预测。📈 关键是最大化这些概率！

Let's start with a quick history lesson to see where Naive Bayes originated before we dive into the nitty gritty details. ⏳
> 让我们从一堂快速的历史课开始，看看朴素贝叶斯的起源，然后再深入研究细节。⏳

## Bayes History 贝叶斯历史
The original Bayes' theorem dates back to the 1700s when Thomas Bayes first described it.
> 最初的贝叶斯定理可以追溯到 1700 年代，当时托马斯·贝叶斯首次描述它。

The theorem provided a way to calculate conditional probabilities.
> 该定理提供了一种计算条件概率的方法。

It laid the foundation for understanding evidence-based statistics and probabilistic reasoning.
> 它为理解循证统计和概率推理奠定了基础。

Over the years, Bayes' theorem became an important tool across fields like economics, medicine, and computing.
> 多年来，贝叶斯定理成为经济学、医学和计算等领域的重要工具。

Fast forward to the 1960s - researchers started extending Bayes for classifying data in machine learning.
> 快进到 1960 年代，研究人员开始扩展贝叶斯，用于对机器学习中的数据进行分类。

But it took high levels of computation to estimate the probabilities needed.
> 但是需要高水平的计算来估计所需的概率。

Then in the 1990s, the "naive" conditional independence assumption dramatically simplified calculations. 💡
> 然后在 1990 年代，“幼稚”的条件独立性假设大大简化了计算。💡

This breakthrough yielded the Naive Bayes classifier algorithm we know and love today! 🥰
> 这一突破产生了我们今天所熟知和喜爱的朴素贝叶斯分类器算法！🥰

Now let's dive into exactly how Naive Bayes works its probabilistic magic! 🎩 ✨
> 现在让我们深入了解朴素贝叶斯是如何发挥其概率魔力的！🎩 ✨

How Naive Bayes Works 朴素贝叶斯的工作原理
The "naive" in Naive Bayes comes from an assumption - all the “features” we use are totally independent from each other! ✋🤚
> 朴素贝叶斯中的“幼稚”来自一个假设——我们使用的所有“特征”都是完全独立的！✋🤚

For example, say we're building a spam filter using words in the email as features. 📧
> 例如，假设我们正在使用电子邮件中的字词作为功能来构建垃圾邮件过滤器。📧

The naive assumption means the word "free" appearing has nothing to do with the word "money" appearing. 💰
> 幼稚的假设意味着“免费”一词的出现与“金钱”一词的出现无关。💰

In the real world, this is often false - spam emails tend to have multiple sketchy words together. 😬
> 在现实世界中，这通常是错误的——垃圾邮件往往将多个粗略的单词放在一起。😬

But it makes the math so much easier! 😅 We just calculate the probability of each word on its own.
> 但它使数学变得如此容易！😅 我们只是单独计算每个单词的概率。

To classify an email as spam or not spam, we:
> 要将电子邮件归类为垃圾邮件或非垃圾邮件，我们：

1️⃣ Find the base rate of spam emails (the prior probability of spam)
> 1️⃣ 找到垃圾邮件的基本率（垃圾邮件的先验概率）

2️⃣ Calculate the probability of each word appearing in spam emails and not spam emails (the likelihoods)
> 2️⃣ 计算每个单词出现在垃圾邮件中而不是垃圾邮件中的概率（可能性）

3️⃣ Use Bayes' theorem to multiply these together and get the **posterior probability** that the email is spam
> 3️⃣ 使用贝叶斯定理将这些相乘，得到电子邮件是垃圾邮件的后验概率

Posterior = Prior x Likelihood1 x Likelihood2 x Likelihood3... 🧮
> 后验 = 先验 x 似然 1 x 似然 2 x 似然 3...🧮

4️⃣ Compare the posterior probability of spam vs not spam
> 4️⃣ 比较垃圾邮件与非垃圾邮件的后验概率

Whichever posterior is higher tells us how to classify the email! 💌
> 无论哪个后部较高，都会告诉我们如何对电子邮件进行分类！💌

So in a nutshell:
> 简而言之：

- Assume feature independence to make the math easy 💪
> - 假设特征独立性，使数学运算变得简单 💪
- Calculate prior and likelihoods across features 📈
> - 计算要素📈的先验和似然
- Multiply to find posterior probabilities 🧮
> - 乘以求后验概率 🧮
- Classify based on the highest posterior! 🏆
> - 根据最高后部进行分类！🏆

## The Algorithm 算法

Let's walk through the key steps of the Naive Bayes algorithm to see the math in action.
> 让我们来看看朴素贝叶斯算法的关键步骤，看看数学的实际应用。

We'll use a simple example trying to classify emails as spam or not spam based on 2 keyword features: contains "free" and contains "money".
> 我们将使用一个简单的示例，尝试根据 2 个关键字特征将电子邮件分类为垃圾邮件或非垃圾邮件：包含“免费”和包含“金钱”。

- **1️⃣ Gather your training data**
> - **1️⃣ 收集训练数据**

We need a training set with emails labeled as spam or not spam to start. Let's say we have 100 emails:
> 我们需要一个训练集，其中包含标记为垃圾邮件或非垃圾邮件的电子邮件才能开始。假设我们有 100 封电子邮件：

- **20 are spam**
> - **20 是垃圾邮件**
- **80 are not spam** 
> - **80 不是垃圾邮件**

- **2️⃣ Calculate the prior probabilities**
> - **2️⃣ 计算先验概率**

The prior probability of an email being spam P(spam) is 20/100 or 0.2
> 电子邮件是垃圾邮件 P（垃圾邮件）的先验概率是 20/100 或 0.2

The prior probability of not spam P(not spam) is 80/100 or 0.8
> 不垃圾邮件 P（非垃圾邮件）的先验概率为 80/100 或 0.8

These are our base rates before seeing any email features.
> 这些是我们在看到任何电子邮件功能之前的基本费率。

- **3️⃣ Calculate the likelihood probabilities**
> - **3️⃣ 计算似然概率**

Let's say in the training data:
> 假设在训练数据中：

15 of the 20 spam emails contain the word "free"
> 20 封垃圾邮件中有 15 封包含“免费”一词
5 of the 80 NOT spam emails contain "free"
> 80 封非垃圾邮件中有 5 封包含“免费”
So the likelihood P("free"|spam) is 15/20 = 0.75
> 所以可能性 P（“free”|spam） 是 15/20 = 0.75

And P("free"|not spam) is 5/80 = 0.0625
> P（“free”|not spam） 为 5/80 = 0.0625

We then do the same for the "money" feature.
> 然后，我们对“金钱”功能执行相同的操作。

- **4️⃣ Multiply likelihoods and prior to get posteriors**
> - **4️⃣ 乘以可能性和获得后验之前**

For an email with "free" and "money", the posterior probabilities are:
> 对于包含“免费”和“金钱”的电子邮件，后验概率为：

P(spam|"free","money") = P(spam) x P("free"|spam) x P("money"|spam)
> P（垃圾邮件|”free“，”money“） = P（垃圾邮件） x P（”free“|垃圾邮件） x P（”money“|垃圾邮件）

P(not spam|"free", "money") = P(not spam) x P("free"|not spam) x P("money"|not spam)
> P（不是垃圾邮件|”free“， ”money“） = P（非垃圾邮件） x P（”free“|非垃圾邮件） x P（”money“|非垃圾邮件）

- **5️⃣ Classify based on highest posterior**
> - **5️⃣ 根据最高后验进行分类**

If P(spam|"free","money") is higher, we classify the email as spam!
> 如果 P（spam|”free“，”money“） 更高，我们将电子邮件归类为垃圾邮件！

## The Advantages 优势

**Fast and simple ⚡️**
> **快速而简单 ⚡️**
- The naive assumption dramatically reduces computation time compared to other algorithms.
> - 与其他算法相比，这种朴素的假设大大减少了计算时间。
- Training is much quicker than neural networks or SVMs.
> - 训练比神经网络或 SVM 快得多。

**Performs well with small data 📊**
> **在处理小数据📊时表现良好**
- Unlike other algorithms, NB doesn't need tons of training data to estimate robust probabilities.
> - 与其他算法不同，NB 不需要大量的训练数据来估计稳健的概率。
- Can learn from fewer examples and still make decent predictions.
> - 可以从较少的例子中学习，并且仍然做出体面的预测。

**Easy to implement 💻**
> **易于实施 💻**
- The math equations are pretty simple to code up.
> - 数学方程式很容易编码。
- Much less programming complexity compared to sophisticated techniques.
> - 与复杂的技术相比，编程的复杂性要低得多。

**Interpretable 🕵️‍♀️** 
> **解释 🕵️ ♀️**
- Since NB relies on conditional probabilities, we can inspect what features have the highest correlations.
> - 由于 NB 依赖于条件概率，因此我们可以检查哪些特征具有最高的相关性。
- More transparent than black box models.
> - 比黑匣子模型更透明。

**Resilient to irrelevant features 💪**
> **对不相关的功能💪具有弹性**
- Adding unnecessary inputs doesn't affect the model too much.  
> - 添加不必要的输入不会对模型产生太大影响。
- Independent probabilities diminish irrelevant relationships.
> - 独立概率减少了不相关的关系。

## Disadvantages of Naive Bayes 朴素贝叶斯的缺点

**Naive assumption 🤨**
> **幼稚的假设 🤨**
- Features are usually not completely independent in real-world data.
> - 特征在实际数据中通常不是完全独立的。
- Violates assumption and leads to inaccurate probabilities.
> - 违反假设并导致不准确的概率。

**Presumes dataset distributions📈**
> **假定数据集分布📈**
- Algorithm presumes data fits standard distribution shapes like Gaussian.
> - 算法假定数据符合标准分布形状，如高斯分布形状。
- Real-world data may not match these expected distributions.
> - 实际数据可能与这些预期分布不匹配。

**Prone to overfitting 🤪** 
> **容易过拟合 🤪**
- With lots of features, easy to overfit to the training data.
> - 具有许多功能，易于过度拟合训练数据。
- Poor generalization to new data. Too many inputs overspecifies.
> - 对新数据的泛化能力差。输入过多，过度指定。

**Metrics difficult to calculate 📉**
> **指标难以计算 📉**
- Standard classification metrics like precision and recall don't apply naturally.
> - 精确率和召回率等标准分类指标并不自然适用。
- Need to use different performance analysis methods.
> - 需要使用不同的性能分析方法。

**Not suitable for complex data 🔮**
> **不适用于复杂数据 🔮**
- Correlated and nonlinear feature relationships violate independence assumption.
> - 相关和非线性特征关系违反了独立性假设。
- Struggles with images, audio, video data.
> - 在图像、音频、视频数据方面苦苦挣扎。

## Applications 应用

**Spam filtering 📧**
> **垃圾邮件过滤 📧**
- Classify emails as spam or not spam based on content features.
> - 根据内容特征将电子邮件分类为垃圾邮件或非垃圾邮件。
- The naive assumption performs well enough here.
> - 朴素的假设在这里表现得足够好。

**Sentiment analysis 😀😡**
> **情绪分析 😀😡**
- Determine positive or negative sentiment of texts like reviews.
> - 确定评论等文本的正面或负面情绪。
- Independent word probabilities work well.
> - 独立词概率效果很好。

**Recommender systems 🛍️**
> **推荐系统 🛍️**
- Recommend products based on past likes/dislikes and product features.
> - 根据过去的好恶和产品功能推荐产品。
- Probabilities help identify preferences.
> - 概率有助于识别偏好。

**Text classification 📑** 
> **文本分类 📑**
- Categorize texts into topics based on word probabilities.
> - 根据单词概率将文本分类为主题。
- Useful for topic modeling and document organizing.
> - 对于主题建模和文档组织很有用。

**Disease prediction 🩺 **
> **疾病预测 🩺**
- Predict presence of disease given diagnostic test outcomes.
> - 根据诊断性测试结果预测疾病的存在。
- Test results can be used as independent features.  
> - 测试结果可用作独立特征。

## TL;DR

- Naive Bayes is a fast, simple classification algorithm that calculates probabilities based on Bayes' theorem and an independence assumption.
> - 朴素贝叶斯是一种快速、简单的分类算法，它根据贝叶斯定理和独立性假设计算概率。
- It performs well on small data where its simplicity is an advantage over more complex methods.
> - 它在小数据上表现良好，与更复杂的方法相比，它的简单性是一个优势。
- Best suited for problems like spam, sentiment, and recommendations where some independence between features exists.
> - 最适合于垃圾邮件、情绪和推荐等功能之间存在一定独立性的问题。
- Not appropriate for complex, correlated data like images, audio, or video.
> - 不适用于复杂的相关数据，如图像、音频或视频。
- Overall, Naive Bayes provides a useful balance of simplicity, speed, and performance!
> - 总体而言，朴素贝叶斯在简单性、速度和性能之间提供了有用的平衡！

## Vocab List️ 词汇表

Bayes' theorem - Defines conditional probability P(A|B) as P(B|A)P(A)/P(B).
> 贝叶斯定理 - 定义条件概率 P（A|B） 作为 P（B|A）P（A）/P（B）。

Likelihood - Probability of data given a hypothesis, P(D|H).
> 似然 - 给定假设的数据概率 P（D|H）。

Prior probability - Initial probability before new evidence, P(H).
> 先验概率 - 新证据之前的初始概率，P（H）。

Posterior probability - Updated probability after new evidence, P(H|D).
> 后验概率 - 新证据后的更新概率，P（H|D).

Conditional independence - Assumption features are unrelated.
> 条件独立性 - 假设功能不相关。

Gaussian distribution - Normal distribution shaped like a bell curve.
> 高斯分布 - 形状像钟形曲线的正态分布。