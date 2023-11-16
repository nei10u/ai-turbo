# Linear Regression
> # 线性回归

Hey friends! 👋 It's me, Miss Neura, here today to break down linear regression. 
> 嘿，朋友们！👋 我，纽拉小姐，今天在这里分解线性回归。


Now I know "regression" sounds kinda intimidating. 😅 But it's really not so bad! 
> 现在我知道“回归”听起来有点吓人。😅 但真的没那么糟糕！


Linear regression is a useful way to model relationships between variables and make predictions. It works by fitting a line to data points. Simple enough, right? 📏
> 线性回归是对变量之间的关系进行建模和进行预测的有用方法。它的工作原理是将一条线拟合到数据点。很简单，对吧？📏


I'll explain step-by-step how linear regression crunches numbers to find the line of best fit. Stick with me! 🤓 
> 我将逐步解释线性回归如何处理数字以找到最佳拟合线。跟我在一起！🤓


By the end, you'll understand the math behind these models and how to build your own. Regression won't seem so scary anymore. 😎
> 最后，您将了解这些模型背后的数学原理以及如何构建自己的模型。回归似乎不再那么可怕了。😎


The key is finding the line that minimizes the distance between predictions and actual data points. We want to squash those residuals! 
> 关键是找到最小化预测和实际数据点之间距离的线。我们想压扁这些残留物！


Let's start with some history to see where linear regression originated before we dig into the details. ⏳
> 让我们从一些历史开始，看看线性回归的起源，然后再深入研究细节。⏳


Then we can walk through an example to really solidify how it works. Alright, let's get our learn on! 🚀
> 然后，我们可以通过一个示例来真正巩固它的工作原理。好了，让我们开始学习吧！🚀


# History # 历史

Linear regression models have been around for a long time! 
> 线性回归模型已经存在了很长时间！


The basics were first described way back in the 1800s. 😮 A mathematician named Francis Galton observed that the heights of parents and their children tended to regress toward the average height in the general population. 
> 早在 1800 年代就首次描述了基础知识。 一位名叫弗朗西斯·高尔顿的数学家观察到，父母和孩子的身高往往会倒退到普通人群的平均身高。 😮


In the early 1900s, more work was done to describe linear regression as we know it today. A famous statistician named Ronald Fisher formalized the model and methods for fitting a line to data.
> 在 1900 年代初期，人们做了更多的工作来描述我们今天所知道的线性回归。一位名叫罗纳德·费舍尔（Ronald Fisher）的著名统计学家正式确定了将线拟合到数据的模型和方法。


The first use of automated computation for linear regression came in the 1940s. A cytologist named Gertrude Cox pioneered the use of punch card machines to calculate regression. Talk about old school tech! 👵
> 线性回归首次使用自动计算是在 1940 年代。一位名叫格特鲁德·考克斯（Gertrude Cox）的细胞学家率先使用穿孔卡机来计算回归。谈谈老派技术！👵


Since then, linear regression has been used across many fields from economics to genetics as computing power exploded over the decades.  
> 从那时起，随着计算能力在几十年的爆炸式增长，线性回归已被用于从经济学到遗传学的许多领域。


Today, it's one of the fundamental machine learning and statistics techniques for modeling linear relationships between variables. Phew, linear regression has come a long way! ⏳
> 如今，它是用于对变量之间的线性关系进行建模的基本机器学习和统计技术之一。呸，线性回归已经走了很长一段路！⏳


Now that we've seen some history, let's move on to understanding how these models actually work their magic. 🧙‍♀️ Onward!
> 现在我们已经了解了一些历史，让我们继续了解这些模型实际上是如何发挥其魔力的。🧙 ♀️ 向前！


# How Linear Regression Works 线性回归的工作原理


Alright, time to dig into the meat of how linear regression works! 🥩
> 好了，是时候深入研究线性回归的工作原理了！🥩


The goal is to model the relationship between two continuous variables - we'll call them X and Y. 
> 目标是对两个连续变量之间的关系进行建模 - 我们称它们为 X 和 Y。


X is the independent variable and Y is the dependent variable. Independent variables are the inputs, dependent variables are the outputs.
> X 是自变量，Y 是因变量。自变量是输入，因变量是输出。


For example, X could be amount spent on ads and Y could be sales generated. Or X could be size of a house and Y could be its price. 
> 例如，X 可以是广告支出，Y 可以是产生的销售额。或者 X 可以是房子的大小，Y 可能是它的价格。


Linear regression finds the best straight line (aka linear model) that fits the data points for X and Y. 
> 线性回归找到拟合 X 和 Y 数据点的最佳直线（又称线性模型）。


This line can be used to predict future values for Y based on a given value of X. Pretty nifty! 
> 这条线可用于根据给定的 X 值预测 Y 的未来值。


The line is characterized by its slope (m) and intercept point (b). Together, they determine where the line is placed.
> 该线的特点是其斜率（m）和截点（b）。它们共同决定了线路的放置位置。


The optimal m and b values are found using the least squares method. What does this mean?
> 使用最小二乘法找到最佳 m 和 b 值。这是什么意思？


It minimizes the distance between data points and the line, also called the residuals. Squashing residuals = happy line. 😊
> 它最小化了数据点和线之间的距离，也称为残差。挤压残差 = 快乐线。😊


This is referred to as the **cost function**, which is a mathematical formula that calculates the total error or "cost" of the current linear regression model. It sums up the residuals (differences between predicted and actual values) for all data points.
> 这称为“成本函数”，它是一个数学公式，用于计算当前线性回归模型的总误差或“成本”。它汇总了所有数据点的残差（预测值和实际值之间的差异）。


So in a nutshell, linear regression uses historical data to find the best fit line for continuous variables. This line can then make predictions! ✨
> 因此，简而言之，线性回归使用历史数据来查找连续变量的最佳拟合线。然后这条线可以做出预测！✨


# The Algorithm # 算法

Let's say we want to model the relationship between the number of hours studied (x) and test score (y).
> 假设我们想要对学习小时数 （x） 和考试分数 （y） 之间的关系进行建模。


We have the following hours studied and test scores:
> 我们有以下学习时间和考试成绩：


Hours Studied (x): 2, 4
> 学习小时数 （x）： 2， 4

Test Scores (y): 80, 90 
> 考试成绩 （y）： 80， 90


Our model is still: 
> 我们的模型仍然是：

**y = mx + b**

We need to find m and b. 
> 我们需要找到 m 和 b。


Let's guess m = 5 and b = 75 
> 让我们猜猜 m = 5 和 b = 75


Plugging this into our model and cost function:
> 将其代入我们的模型和成本函数中：


For x = 2 hours studied, predicted test score would be y = 5x2 + 75 = 85 
> 对于 x = 2 小时的学习，预测测试分数为 y = 5x2 + 75 = 85

Actual y = 80 实际 y = 80
Residual = 85 - 80 = 5
> 残差 = 85 - 80 = 5


Now let's test x = 4 hours studied, predicted y = 5x4 + 75 = 95
> 现在让我们检验 x = 4 小时的研究时间，预测 y = 5x4 + 75 = 95

Actual y = 90  
> 实际 y = 90

Residual = 95 - 90 = 5 
> 残差 = 95 - 90 = 5


Cost function:  
> 成本函数：
The formula to calculate the cost function (or the error between the predicted vs the actual across data points) is **J(m,b) = Σ(y - (mx + b))^2**
> 计算成本函数（或跨数据点的预测值与实际值之间的误差）的公式为 **J（m，b） = Σ（y - （mx + b））^2**


So in our example: 
> 因此，在我们的示例中：
J(m,b) = 5^2 + 5^2 = 50
J（米，b） = 5^2 + 5^2 = 50


To minimize J, we can tweak m and b. Let's try:
> 为了最小化 J，我们可以调整 m 和 b。让我们试试：

m = 10, b = 60
m = 10， b = 60


Now: 
> 现在：
For x = 2 hours studied, predicted y = 10x2 + 60 = 80  
> 对于 x = 2 小时的研究，预测 y = 10x2 + 60 = 80

Residual = 0 
> 残差 = 0

For x = 4, predicted y = 10x4 + 60 = 100
> 对于 x = 4，预测值 y = 10x4 + 60 = 100

Residual = 10 
> 残差 = 10

Cost function:  
> 成本函数：
J(m,b) = 0^2 + 10^2 = 100
J（米，b） = 0^2 + 10^2 = 100


So we can see tweaking m and b changes the residuals and cost J. We want to find the combo with lowest J.
> 因此，我们可以看到调整 m 和 b 会改变残差和成本 J。我们想找到 J 最低的组合。


In this tiny example, m=10, b=60 minimizes J. 
> 在这个小例子中，m=10，b=60 最小化 J。


So our final line is:
> 所以我们的最后一行是：

**y = 10x + 60**

The equation y = 10x + 60 represents the best fit straight line relationship between hours studied (x) and test scores (y).
> 方程 y = 10x + 60 表示学习时间 （x） 和考试分数 （y） 之间的最佳拟合直线关系。


Specifically: 具体说来：

- The slope is 10. This means for every 1 additional hour studied, the model predicts the test score will increase by 10 points.
> - 斜率为10。这意味着每多研究 1 小时，模型预测测试分数将增加 10 分。

- The intercept is 60. This means when 0 hours are studied, the model predicts a test score of 60 points.
> - 截距为 60。这意味着当研究 0 小时时，模型预测测试分数为 60 分。


So in plain terms: 所以简单来说：

- Studying more hours corresponds to getting higher test scores
> - 学习时间越长，考试成绩就越高

- There is a linear relationship where studying 10 additional hours predicts scoring 10 more points
> - 存在线性关系，即多学习 10 小时预示着多得 10 分

- Even with 0 hours studied, the model predicts a baseline score of 60 points
> - 即使研究了 0 小时，该模型也预测基线分数为 60 分


This simple linear model captures the positive correlation between study time and test results. The line models how test scores are expected to improve by a certain amount per hour studied.
> 这个简单的线性模型捕捉了研究时间和测试结果之间的正相关关系。该线模拟了每研究一小时，考试成绩有望提高一定量。


# Advantages # 优势

Alright, linear regression has some really great advantages worth highlighting! 😃
> 好吧，线性回归有一些非常大的优点值得强调！😃


First up, it's super simple to implement and interpret. ♟️ The math isn't too wild, and the output is an easy-to-understand line equation.
> 首先，它的实现和解释非常简单。♟️ 数学不是太疯狂，输出是一个易于理解的线方程。


It can also handle multiple independent variables at once! 🤯 Just plug them into the equation and away you go. Multi-variable modeling ftw!
> 它还可以同时处理多个自变量！🤯 只需将它们代入方程式中，就可以了。多变量建模ftw！


Additionally, linear regression works well when there truly is a linear relationship in the data. 📈 It shines when x and y are correlated in a straight-ish line.
> 此外，当数据中确实存在线性关系时，线性回归效果很好。📈 当 x 和 y 在一条直线上相关时，它会发光。


From a computational side, linear regression has pretty minimal requirements. 💻 It can run efficiently without massive processing power.
> 从计算方面来看，线性回归的要求非常低。💻 它可以在没有大量处理能力的情况下高效运行。


Lastly, this technique is used across so many fields and has stood the test of time. 🏆 A true flexible MVP algorithm!
> 最后，这种技术被用于许多领域，并且经受住了时间的考验。🏆 真正灵活的MVP算法！


Of course, every hero has their kryptonite. Let's flip to some disadvantages next...😼
> 当然，每个英雄都有自己的氪石。接下来让我们来看看一些缺点......😼


# Disadvantages # 缺点

Alright, linear regression does have some weaknesses we gotta talk about. ⚠️
> 好吧，线性回归确实有一些我们必须讨论的弱点。⚠️


First, it relies on the assumption that the relationship between variables is actually linear. 📏 Messy non-linear data will throw things off. 
> 首先，它依赖于变量之间的关系实际上是线性的假设。📏 杂乱无章的非线性数据会让事情变得糟糕。


It's also prone to overfitting with lots of input features, which can lead to poor generalization on new data. Too much complexity can skew the model. 🤪
> 它还容易对大量输入特征进行过度拟合，这可能导致对新数据的泛化性不佳。过于复杂可能会使模型出现偏差。🤪


Linear regression is sensitive to outliers too. A few weird data points can drastically change that best fit line. 👀
> 线性回归对异常值也很敏感。一些奇怪的数据点可以彻底改变最佳拟合线。👀


And if there's correlation but not causation between variables, the model might make inaccurate predictions. 🤥 
> 如果变量之间存在相关性但没有因果关系，则模型可能会做出不准确的预测。🤥


Finally, linear regression can't capture non-linear relationships that are inherently more complex. 🤷‍♀️ Curves, loops, jumps - nope!
> 最后，线性回归无法捕获本质上更复杂的非线性关系。🤷 ♀️ 曲线、循环、跳跃 - 不！


But don't fret! Many of these disadvantages have workarounds or alternative algorithms. 💪
> 但不要担心！其中许多缺点都有变通方法或替代算法。💪


Alright, we're nearing the finish line! Let's talk about some real-world applications next. 🏁
> 好了，我们快到终点线了！接下来我们来谈谈一些实际应用。🏁


# Applications # 应用

One of the most popular uses of linear regression is predicting housing prices! 🏡 
> 线性回归最流行的用途之一是预测房价！🏡


Features like square footage, location, age, etc. are used to model price. This helps estimate the value of new properties.
> 平方英尺、位置、年龄等特征用于对价格进行建模。这有助于估计新属性的价值。


It's also commonly used in finance to forecast things like revenue, demand, inventory needs and other trends. 💰 Helpful for budgeting!
> 它也常用于金融领域，用于预测收入、需求、库存需求和其他趋势。💰 有助于预算！


Economics is another field that leverages linear regression. It can estimate impacts of policy changes on metrics like GDP, unemployment, inflation, and growth. 📈 
> 经济学是另一个利用线性回归的领域。它可以估计政策变化对 GDP、失业率、通货膨胀和增长等指标的影响。📈


Insurance companies use these models to assess risk factors and set appropriate premiums. Predicting claims helps pricing. 🚗
> 保险公司使用这些模型来评估风险因素并设定适当的保费。预测索赔有助于定价。🚗


Even fields like healthcare apply linear regression. It helps model the effect of medications, treatments, diets on patient outcomes. 🩺 
> 甚至像医疗保健这样的领域也应用线性回归。它有助于模拟药物、治疗、饮食对患者预后的影响。🩺


Beyond these examples, linear regression powers many fundamental machine learning algorithms under the hood too!
> 除了这些例子之外，线性回归还为许多基本的机器学习算法提供了动力！


It provides a simple baseline for modeling all kinds of relationships between variables. 🔍
> 它为对变量之间的各种关系进行建模提供了一个简单的基线。🔍


# TL;DR

- Linear regression is used to model the relationship between two continuous variables. It fits a straight line through data points by minimizing the residuals (differences between predicted and actual y values).
> - 线性回归用于对两个连续变量之间的关系进行建模。它通过最小化残差（预测值和实际 y 值之间的差异）来拟合数据点的直线。

- The line equation is defined as y=mx+b, where m is the slope and b is the intercept. The slope tells us how much y changes for each unit increase in x. The intercept is where the line crosses the y-axis.
> - 直线方程定义为 y=mx+b，其中 m 是斜率，b 是截距。斜率告诉我们 x 每增加一个单位，y 就会变化多少。截距是直线与 y 轴相交的位置。

- The optimal m and b values are found via gradient descent - iteratively tweaking to reduce error. This results in the line of best fit that most closely models the linear relationship.
> - 通过梯度下降找到最佳 m 和 b 值 - 迭代调整以减少误差。这样可以生成最接近线性关系建模的最佳拟合线。

- Key strengths of linear regression include simplicity, interpretability, and modeling linear correlations well. Weaknesses include sensitivity to outliers and inability to capture non-linear trends.
> - 线性回归的主要优势包括简单性、可解释性和很好地模拟线性相关性。弱点包括对异常值的敏感性和无法捕捉非线性趋势。


Overall, linear regression is a fundamental machine learning technique for predicting a numerical target based on linear relationships between variables. It continues to serve as a building block for more advanced algorithms too!
> 总体而言，线性回归是一种基本的机器学习技术，用于根据变量之间的线性关系预测数值目标。它也继续作为更高级算法的构建块！

# Vocab List # 词汇表

- Regression - Modeling the relationship between variables
> - 回归 - 对变量之间的关系进行建模

- Dependent variable - The output variable we're predicting (y)
> - 因变量 - 我们预测的输出变量 （y）

- Independent variable - The input variable (x)
> - 自变量 - 输入变量 （x）

- Slope (m) - How much y changes per change in x
> - 斜率 （m） - 每次 x 变化的 y 变化量

- Intercept (b) - Where the line crosses the y-axis
> - 截距 （b） - 直线与 y 轴相交的位置

- Residuals - The differences between predicted and actual y
> - 残差 - 预测 y 和实际 y 之间的差异

- Cost function (J) - Quantifies total error of the line
> - 成本函数 （J） - 量化生产线的总误差

- Gradient descent - Iteratively updates to minimize J
> - 梯度下降 - 迭代更新以最小化 J

- Overfitting - Model matches training data too closely
> - 过拟合 - 模型与训练数据过于匹配