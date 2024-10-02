const projects = {
  post1: {
    id: 1,
    title: "Finance Portfolio Optimization Methods",
    synopsis:
      "This project explores various portfolio optimization methods in finance",
    content: `
        <p>Exploring different portfolio optimization methods on a control set of securities</p>

        <h3>Table of Contents</h3>
        <ul>
            <li><a href="#dataset">Dataset</a></li>
            <li><a href="#exploratory-data-analysis">Exploratory Data Analysis</a></li>
            <li><a href="#risk-parity">Risk Parity</a></li>
            <li><a href="#mean-variance">Mean-Variance</a></li>
            <li><a href="#genetic-algorithms">Genetic Algorithms</a></li>
            <li><a href="#future-work">Future Work</a></li>
        </ul>

        <h3 id="dataset">Dataset</h3>
        <p>In forming the list of control securities, I wanted to ensure we diversify across asset classes such as stocks, bonds, commodities, real estate, etc. The simplest way to do so was to use ETFs such as SPY, which can represent asset classes.</p>
        <p>The dataset is daily adjusted closing prices for the securities below. It is collected from Yahoo Finance using the Python package 'yfinance' and ranges from 2010-01-01 to 2023-01-01.</p>
        
        <h4>Securities</h4>
        <ul>
            <li>Large-Cap US Equity: SPDR S&P 500 ETF (SPY)</li>
            <li>US Treasury Bonds: iShares 7-10 Year Treasury Bond ETF (IEF)</li>
            <li>Corporate Bonds: iShares iBoxx $ Investment Grade Corporate Bond ETF (LQD)</li>
            <li>Oil: United States Oil Fund (USO)</li>
            <li>Gold: SPDR Gold Shares (GLD)</li>
            <li>US Real Estate: Vanguard Real Estate ETF (VNQ)</li>
            <li>Global Real Estate: SPDR Dow Jones Global Real Estate ETF (RWO)</li>
            <li>Small-Cap Stocks: iShares Russell 2000 ETF (IWM)</li>
            <li>High Yield Bonds: iShares iBoxx $ High Yield Corporate Bond ETF (HYG)</li>
            <li>Broad Commodities: Invesco DB Commodity Index Tracking Fund (DBC)</li>
            <li>European Markets: Vanguard FTSE Europe ETF (VGK)</li>
        </ul>

        <h3 id="exploratory-data-analysis">Exploratory Data Analysis</h3>
        <p>Before diving into the optimization methods, I wanted to do some high-level EDA on our data.</p>
        <p>In the correlation chart below, we see how our securities relate to each other. The main observation here is that most of the securities have a positive correlation, except commodities like USO, GLD, and DBC. Introducing such securities to the portfolio is theoretically good for the overall risk profile.</p>
        <img src="assets/data_corr.png" alt="Securities Correlation Heatmap">
        <p>We can also plot the prices of each security over the time frame to give insights into the risk and return profiles of them individually. For example, below we see that SPY offers high returns in the period but is more volatile (relatively):</p>
        <img src="assets/time_series.png" alt="Time Series of Security Prices">
        <p>Normalizing the prices improves chart readability:</p>
        <img src="assets/normalized_tis.png" alt="Normalized Time Series of Security Prices">

        <h3 id="risk-parity">Risk Parity</h3>
        <p>The theory behind Risk Parity Optimization is to purchase securities such that each one contributes an equal amount of risk to the portfolio. This can be achieved generally by purchasing less of the more risky securities and vice versa. Or mathematically:</p>
        <img src="assets/parity_forms.png" alt="Parity Formula">
        <p>So with that, we implement the simple algorithm in Python, using scipy.minimize to minimize the difference in risk contribution from each asset.</p>
        <p>We find the following results:</p>
        <ul>
            <li>Annualized Expected Portfolio Return: 4.57%</li>
            <li>Annualized Portfolio Risk (Volatility): 6.87%</li>
        </ul>
        <p>This is an underwhelming result, but it is not surprising as we have some low volatility, low return securities in the portfolio which we would expect to get a lot of the capital distribution from this method.</p>
        <p>Indeed we see below that over 50% of capital is in Government and Corporate Bonds, as they have low risk profiles. The problem with this is that we don't expose ourselves to enough potential returns (in my opinion).</p>
        <img src="assets/rp_pie.png" alt="Risk Parity Weights">
        <p>Below are the cumulative returns of all the securities and the risk parity portfolio. I takeaway that this is a good strategy for someone who is quite risk averse, but still wants the benefits of diversification, rather than just buying, say, U.S. Treasuries.</p>
        <img src="assets/rp_returns.png" alt="Risk Parity Returns">
        <p>I also think this would be an interesting method to apply to just the stock market, that way you could see some higher returns. For example, instead of using the standard S&P 500 Index Fund, you could weight stocks according to risk parity.</p>

        <h3 id="mean-variance">Mean-Variance</h3>
        <p>My reference for this method: <a href="https://www.columbia.edu/~mh2078/FoundationsFE/MeanVariance-CAPM.pdf" target="_blank">View Mean-Variance CAPM PDF</a></p>
        <p>In reading about Markowitz's mean-variance method and the CAPM model, I was able to gain the most intuition from the mathematical formulas and graphs, so I'll lean on those here.</p>
        <p>Also note that I don't consider our list of securities to contain a risk-free asset. We assume that we have n risky which make up the return vector 𝑅. We also assume that 𝑅 follows a Multivariate Normal distribution with mean vector 𝜇 and covariance matrix 𝛴.</p>
        <p>The mean-variance portfolio optimization problem is formulated as:</p>
        <img src="assets/contraint.png" alt="Budget Constraint">
        <p>(Taken from above lecture notes)</p>
        <p>In English, this lays out the optimization problem as trying to find the weights (w) that produce the minimum risk (portfolio variance), given a target return (p). We also set constraints in implementation: The sum of the weights is equal to 1 (all capital is invested) and each weight ∈ [0,1] (no options or leveraging).</p>
        <p>Given this, upon implementation, we can give the algorithm a target return rate and output the portfolio with the lowest risk. Using this, I derive both the portfolio with the highest Sharpe ratio and the efficient portfolios frontier for our </p>
        <p>In the below table are data for various portfolios across the efficient frontier. Note that securities not listed are weighted at 0.</p>
        <table>
            <tr>
                <th>Portfolio</th>
                <th>ETF</th>
                <th>Weight (%)</th>
                <th>Return</th>
                <th>Volatility</th>
            </tr>
            <tr>
                <td>Highest Sharpe Ratio&nbsp&nbsp</td>
                <td>SPY</td>
                <td>32.15</td>
                <td>0.06</td>
                <td>0.06</td>
                 </tr>
            <tr>
                <td></td>
                <td>IEF</td>
                <td>66.26</td>
                <td></td>
                <td></td>
            </tr>
            <tr>
                <td></td>
                <td>GLD</td>
                <td>1.59</td>
                <td></td>
                <td></td>
            </tr>
            <tr>
                <td>Portfolio 50</td>
                <td>IEF</td>
                <td>22.69</td>
                <td>-0.03</td>
                <td>0.28</td>
            </tr>
            <tr>
                <td></td>
                <td>USO</td>
                <td>77.31</td>
                <td></td>
                <td></td>
            </tr>
            <tr>
                <td>Portfolio 150</td>
                <td>IEF</td>
                <td>69.00</td>
                <td>0.00</td>
                <td>0.11</td>
            </tr>
            <tr>
                <td></td>
                <td>USO</td>
                <td>31.00</td>
                <td></td>
                <td></td>
            </tr>
            <tr>
                <td>Portfolio 250</td>
                <td>SPY</td>
                <td>7.95</td>
                <td>0.04</td>
                <td>0.05</td>
            </tr>
            <tr>
                <td
                    </td>
                <td>IEF</td>
                <td>66.52</td>
                <td></td>
                <td></td>
            </tr>
            <tr>
                <td></td>
                <td>HYG</td>
                <td>21.45</td>
                <td></td>
                <td></td>
            </tr>
            <tr>
                <td></td>
                <td>DBC</td>
                <td>4.08</td>
                <td></td>
                <td></td>
            </tr>
            <tr>
                <td>Portfolio 350</td>
                <td>SPY</td>
                <td>46.54</td>
                <td>0.07</td>
                <td>0.08</td>
            </tr>
            <tr>
                <td></td>
                <td>IEF</td>
                <td>50.94</td>
                <td></td>
                <td></td>
            </tr>
            <tr>
                <td></td>
                <td>GLD</td>
                <td>2.52</td>
                <td></td>
                <td></td>
            </tr>
            <tr>
                <td>Portfolio 450</td>
                <td>SPY</td>
                <td>81.41</td>
                <td>0.11</td>
                <td>0.15</td>
            </tr>
            <tr>
                <td></td>
                <td>IEF</td>
                <td>13.58</td>
                <td></td>
                <td></td>
            </tr>
            <tr>
                <td></td>
                <td>GLD</td>
                <td>5.01</td>
                <td></td>
                <td></td>
            </tr>
            <tr>
                <td>Portfolio 500</td>
                <td>SPY</td>
                <td>100.00</td>
                <td>0.13</td>
                <td>0.18</td>
            </tr>
        </table>

        <p>My takeaway from this is that the mean variance algorithm is more flexible for investors than risk parity (above). We see that higher return portfolios are weighted in favor of riskier  with the most extreme example being all capital invested in SPY, which had the highest mean returns of all securities at hand. So, even though the max Sharpe ratio may suggest the investor to be less risky, there are alternative suggestions for someone looking for more return.</p>
        <p>The efficient portfolio frontier shows the entire spectrum of efficient portfolios, it follows that with higher returns come higher risk. Finally, I draw a line at the minimum risk, as all portfolios below it would not be invested in by a rational person (they can gain higher returns for the same risk).</p>
        <img src="assets/mv_frontier.png" alt="Efficient Portfolios Frontier">

        <h3 id="genetic-algorithms">Genetic Algorithms</h3>
        <p>Reference for genetic algorithms: <a href="chrome-extension://bdfcnmeidppjeaggnmidamkiddifkdib/viewer.html?file=https://www.graham-kendall.com/papers/sgk2005.pdf" target="_blank">View Mean-Variance CAPM PDF</a></p>
        <p>Genetic algorithms, generally, are meant to replicate evolutionary biology and natural selection. In our case, we'll implement them to optimize a portfolio of securities.</p>

        <h4>Algorithm Description</h4>
        <p>First, we initialize a population of n portfolios with random weights. Over m generations, we create new populations based off prior portfolios with the highest level of fit. In this case, fitness is determined by maximizing the Sharpe ratio, as used in the mean-variance optimization. Portfolios in a given population with higher levels of fit will be assigned higher probabilities of 'reproduction':</p>
        <img src="assets/prob_fitness.png" alt="Fitness Probability">
        <p>where:</p>
        <p>𝑃𝑖 is the selection probability of individual 𝑖</p>
        <p>𝑓𝑖 is the fitness of individual 𝑖</p>
        <p>𝑁 is the total number of individuals in the population.</p>
        <p>Portfolios selected by their probability will then proceed into crossover. In this, we simulate reproduction by combining the weights of two parent portfolios around some random pivot index. Each 'couple' will produce two 'children' consisting of their weights arrays before and after the pivot index.</p>
        <p>We also introduce mutation with a mutation rate of 0.01. Individuals will be selected for mutation at random based on this rate, and have one of their weights altered randomly.</p>
        <p>So, with this, we initialize <code>population_size=200</code> and <code>num_generations=10000</code>, and run the algorithm. We expect to see the fitness increase over the course of generations, thus generating a better portfolio.</p>

        <h4>Optimal Portfolio Weights</h4>
        <table>
            <tr>
                <th>ETF</th>
                <th>Weight (%)</th>
            </tr>
            <tr>
                <td>SPY</td>
                <td>54.28</td>
            </tr>
            <tr>
                <td>VGK</td>
                <td>0.00</td>
            </tr>
            <tr>
                <td>IEF</td>
                <td>30.37</td>
            </tr>
            <tr>
                <td>LQD</td>
                <td>3.05</td>
            </tr>
            <tr>
                <td>USO</td>
                <td>0.02</td>
            </tr>
            <tr>
                <td>GLD</td>
                <td>2.80</td>
            </tr>
            <tr>
                <td>VNQ</td>
                <td>0.53</td>
            </tr>
            <tr>
                <td>RWO</td>
                <td>0.05</td>
            </tr>
            <tr>
                <td>IWM</td>
                <td>0.35</td>
            </tr>
            <tr>
                <td>HYG</td>
                <td>8.53</td>
            </tr>
            <tr>
                <td>DBC</td>
                <td>0.00</td>
            </tr>
        </table>

        <h4>Final Portfolio Performance</h4>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Return</td>
                <td>0.0854</td>
            </tr>
            <tr>
                <td>Volatility&nbsp</td>
                <td>0.1001</td>
            </tr>
        </table>

        <p>Below is the weights in a bar chart:</p>
        <img src="assets/genetic_weights.png" alt="Genetic Weights">

        <p>We also see that the fitness trends upwards over generations, indicating our algorithm is working correctly.</p>
        <img src="assets/genetic_evolution.png" alt="Genetic Evolution">

        <p>An interesting note on this algorithm is that each run outputs a different set of weights. Because of the random nature, and the fact that we aren't quite achieving the max Sharpe ratio. This leads to much variability in returns and risk from run to run. Which makes me wonder, how would finance professionals actually implement the algorithm, and how they would finally select a portfolio?</p>

        <h3 id="future-work">Future Work</h3>

        <h4>Expanding the Dataset</h4>
        <p>The current dataset spans from 2010-01-01 to 2023-01-01 and includes a diverse set of asset classes. Future work could involve:</p>
        <ul>
            <li><strong>Incorporating More Securities</strong>: Including individual stocks could provide a more granular view of the market.</li>
            <li><strong>Extending the Time Frame</strong>: Analyzing a longer historical period could offer more insights into the performance of various optimization methods over different market cycles.</li>
        </ul>

        <h4>Real-World Constraints</h4>
        <p>To make the portfolio optimization more applicable to real-world scenarios, future work should consider:</p>
        <ul>
            <li><strong>Transaction Costs</strong>: Including transaction costs and other fees in the optimization process to better simulate real trading environments.</li>
            <li><strong>Regulatory Requirements</strong>: Adhering to regulatory constraints, such as diversification rules, which might affect institutional investors.</li>
        </ul>`,
    link: "https://github.com/robbyhooker/Portfolio_Optimization",
  },
  post2: {
    id: 2,
    title: "Classification for Conservation",
    synopsis:
      "An exploration of machine learning applied to ecology and wildlife conservation",
    content: `
    <p>This project is an exploration of machine learning applied to ecology and wildlife conservation. The specific aim of the project was to detect, classify, and track wildlife in aerial imagery. As a byproduct, this presented me with the opportunity to learn about machine learning generally, and in its current state-of-the-art. To achieve the desired model, I explore existing research, methods, and data related to the subject. Furthermore, we will explore the famous real time detection algorithm YOLO, discussing how it works internally, why it has become so prevalent, and its application to our specific case of aerial wildlife imagery.</p>

    <p>The inspiration for this project comes from my love/fascination for nature and interest in machine learning. The reason I chose to study computer science is because the breadth of applications the discipline presents. The real time object detection of wildlife is both practical for conservation efforts, and insightful into computer vision research, making this a compelling endeavor. In the field, models like the proposed can be used to monitor wildlife habitats, assisting in population censuses, identifying poachers, and ecosystem analysis. On a personal level, I hope the commencement of this project is the beginning of my further exploration of these fields.</p>

    <p>This project is a machine learning model which performs object detection on aerial wildlife image data, which is being done using the YOLOv8 algorithm. YOLOv8: [Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLO (Version 8.0.0) [Computer software]. <a href="https://github.com/ultralytics/ultralytics">https://github.com/ultralytics/ultralytics</a></p>

    <h2>Table of Contents</h2>
    <ol>
        <li><a href="#dataset">Dataset</a></li>
        <li><a href="#methods">Methods</a></li>
        <li><a href="#results">Results</a></li>
        <li><a href="#conclusions">Conclusions</a></li>
        <li><a href="#future-work">Future Work</a></li>
    </ol>

    <h2 id="dataset">Dataset</h2>
    <p>That data used is from the WAID dataset, whose accompanying paper can be found at the following link: <a href="https://www.mdpi.com/2076-3417/13/18/10397">https://www.mdpi.com/2076-3417/13/18/10397</a></p>
    <p>In training machine learning models, the two most valuable to an engineer are data and compute. Prior to training the wildlife classification model at hand, these resources were carefully explored. The WAID1 dataset is a free and publicly available dataset designed for the object detection of wildlife in aerial imagery. It contains 14,366 images of wildlife, split into train, validation, and test sets at a 7:2:1 ratio, respectively. For each image in the set there is an accompanying .txt file which contains the class of each animal in the image, along with their bounding box coordinates. This data format is known as “YOLO” format, as it is used for the ever-popular real time object detection algorithm, YOLO (You Only Look Once). The images in this set are also a uniform 640 x 640 pixels, which is the default input for YOLOv8, which was the state-of-the-art version during the training of this model. Several other datasets were considered; however, it was determined that WAID was the most comprehensive and usable set of those available. The animals included in the data are cattle, sheep, kiang, camels, seals, and zebras.</p>
    
    <p>Example Image of Sheep:</p>
    <img src="assets/sheeptest.jpg" alt="Example Image of Sheep"><br>

    <h2 id="methods">Methods</h2>
    <p>The reason compute is so essential in training machine learning algorithms is because the models may need to perform billions, or even trillions of floating-point operations to adjust the weights of the model. Performing this number of operations can be very time consuming, however the use of a dedicated graphics processing unit (GPU) can greatly increase performance. This is because GPUs are tailored to perform matrix operations, such as convolutions and multiplications, in parallel, allowing them to do so much faster than the CPU. Knowing this, when training a large convolutional neural network, it is essential to utilize a GPU. With recent demand for cloud computing, cloud GPUs have become increasingly available at lower costs. For this reason, in training our model, we will deploy the Google Collaboratory's cloud GPU infrastructure, which will increase performance on the order of 100x!</p>
    
    <p>The algorithm chosen for the detection problem was YOLOv8. YOLOv8 uses deep learning to build a convolutional neural network which makes real time object detections. This was an obvious choice for three main reasons, the first being the ease of use with our chosen dataset, as it is already in the correct format for YOLOv8. Next, this model will be most useful in wildlife conservation efforts if it is able to detect in real time, and YOLO is built for real time detection. Finally, the popularity of the algorithm also means it has a large support community, and an abundance of resources at the engineer's disposal.</p>
    
    <p>The algorithm gets its name from the fact that it can detect objects in a single forward pass of a given image. This breakthrough was essential to the high efficiency of the algorithm, allowing it to detect in real time. The diagram below is taken from the original YOLO research paper. Of course, YOLOv8 has become far more complex than the original, however this image is still relevant for high level understanding of the algorithm. The convolution layers of the algorithm downsize the input image while performing various convolution operations, each of which produces a feature map. In theory and practice, these feature maps contain key information such as edges, corners, patterns, gradients, etc. These convolution layers along with activation and pooling layers eventually shrink the input image to an output tensor. This tensor contains the algorithm's prediction for the classes of the objects detected, along with their corresponding bounding boxes.</p>

    <p>In the case of our 640x640 pixel images, the model will eventually downsize them into a final 20x20x16 tensor. Each cell in the tensor contains the algorithm's overall prediction for the image. This includes the confidence score of an object being present in the cell, the bounding box coordinate prediction, and a probability assigned to each class.</p>
    
    <img src="assets/yolo.png" alt="YOLO Algorithm Diagram"><br>

    <p>Originally, these predictions are random, as the model has no sense of the truth. However, over each epoch, a proper machine learning model will decrease its error and become a sufficient predictor. It does so by comparing its output prediction to the ground truth classes and bounding boxes that correspond with the image in our dataset. The algorithm calculates the error between its prediction and the truth and uses it to adjust the parameters that determine the convolution function. This is known as backpropagation. As the parameters become better at predicting the contents of the images, the model becomes learned.</p>

    <p>To measure the performance of the model, we will use mean average precision (mAP). This is calculated by the area under the curve of the precision-recall curve, which is shown below.</p>
    
    <img src="assets/pr_curve.png" alt="Precision-Recall Curve">

    <h2 id="results">Results</h2>
    <p>For the training of our algorithm on aerial wildlife imagery, we will use the methods discussed, and expect to see a decrease in error and increase in accuracy over the course of training. In this case, we will train over 100 passes of the dataset (epochs), in each epoch the model will ‘see’ each image in the dataset once, adjusting its parameters accordingly for each batch of images. The results of this project were obtained using a batch size of 16, over 100 epochs. The charts below show the error and accuracy statistics of our model over the course of training.</p>
    
    <p>The box_loss measures the error in the model’s bounding box predictions over the training of the model. It is calculated by the mean squared error between detected box coordinates and corresponding ground truth coordinates. Similarly, cls_loss measures the error in the model’s class predictions over the training of the model. It is calculated by the cross-entropy loss between the predicted class probabilities and the actual class label. Considering these are measures of error, we are pleased to see them decrease throughout training. Similarly, since mAP50 and mAP50-95 are measures of accuracy, we are pleased to see them increase over the course of training.</p>
    
    <img src="assets/Screenshot%202024-04-20%20111331.png" alt="Training Error and Accuracy">
    <img src="assets/Screenshot%202024-04-20%20111356.png" alt="Training Error and Accuracy">

    <p>We also see the model is a sufficient object detector in the test images:</p>
    <img src="assets/sheep_annotated.jpg" alt="Test Image with Annotations">

    <h2 id="conclusions">Conclusions</h2>
    <p>Upon testing the model against data unseen in the training process, the statistics remain promising. We see that precision, recall, and map50 are consistently above 90%. The mAP50-95 averages 63.7% across all classes, which is still impressive considering the high IOU threshold. Given these statistics, I can confidently say this model is an adequate classifier of wildlife in images. It is also proven to be a sufficient tracker when applied to real world video of wildlife! So, the model is not only working on paper, but also performs well in practical cases. This is an exciting outcome for me, as I have learned how to apply methods that are being used to further the fields of ecology research and wildlife conservation!</p>

    <h2 id="future-work">Future Work</h2>
    <p>Going forward, I would like to bolster this model by adding more data, including more species of animals. Doing so might entail traveling to a habitat and collecting new data to add to the dataset. The goal is to have a model that can make predictions on all classes of animals in a habitat, and then deploy the model via live drone or UAV imagery. Coupling this with a dedicated GPU would allow for real time detection of animals in their natural environment!</p>
    
    <p>I also look forward to tweaking the hyperparameters of the network and analyzing its inner layers. This could lead to insight and a higher performing model.</p>`,
    link: "https://github.com/robbyhooker/Classication_Conservation",
  },
  post3: {
    id: 3,
    title: "Formula 1 Exploratory",
    synopsis:
      "This project involves exploratory data analysis (EDA) on historical Formula 1 race data",
    content: ` <h2>Overview</h2>
            <p>This project involves exploratory data analysis (EDA) on historical Formula 1 race data, focusing on understanding trends and patterns among teams, drivers, and race characteristics over the years.</p>

            <h3>Dataset</h3>
            <p>The dataset used can be found at the following link: <a href="https://www.kaggle.com/datasets/lakshayjain611/f1-races-results-dataset-1950-to-2024/data">Kaggle</a>.</p>
            <p>It contains:</p>
            <ul>
                <li><strong>Total Rows:</strong> 1,110</li>
                <li><strong>Columns:</strong> Grand Prix, Date, Winner, Car, Laps, Time, Name Code</li>
                <li><strong>Time Range:</strong> Covers races from 1950 onwards</li>
                <li><strong>Key Features:</strong>
                    <ul>
                        <li><strong>Grand Prix:</strong> The name of the race event</li>
                        <li><strong>Date:</strong> Date of the race</li>
                        <li><strong>Winner:</strong> The name of the winning driver</li>
                        <li><strong>Car:</strong> The car/constructor used by the winner</li>
                        <li><strong>Laps:</strong> Number of laps completed in the race</li>
                        <li><strong>Time:</strong> The total time taken to win the race</li>
                    </ul>
                </li>
            </ul>

            <h3>Analyses</h3>

            <h3>1. Cumulative Wins Over Time</h3>
            <p><strong>Objective:</strong> Track the progression of cumulative wins for teams and drivers.</p>
            <p><strong>Visualization:</strong> Line plots show the accumulation of wins, highlighting periods of dominance.<br><br> We see in the cumulative team wins chart that Scuderia Ferrari have always been the top team in racing.<br>However, with recent struggles and epic runs by Red Bull and Mercedes, they have entered the conversation, along with McLaren.</p>
            <div class="charts">
                <img src="assets/Charts/cum_teams.png" alt="Cumulative Team Wins">
            </div>
            <p>Based on driver wins over time, it is hard to deny that Lewis Hamilton is the greatest racer of all time.<br>Max Verstappen's astonishing trajectory may indicate that he'll threaten this title.</p>
            <div class="charts">
                <img src="assets/Charts/cum_drivers.png" alt="Cumulative Driver Wins">
            </div>

            <h3>2. Dominance Periods</h3>
            <p><strong>Objective:</strong> Identify specific periods when teams or drivers were dominant.</p>
            <p><strong>Visualization:</strong> Time-dependent bar plot shows dominant eras for teams and drivers.<br><br> Team dominance shows how Ferrari got their lead in the wins category, but also exposes their recent absence.<br>Hopefully Leclerc and Sainz can be catalysts of change in 2025.</p>
            <div class="charts">
                <img src="assets/Charts/dom_teams.png" alt="Dominant Teams">
            </div>
            <p>Driver dominance periods further demonstrate Verstappen and Hamilton's recent grip on the sport.</p>
            <div class="charts">
                <img src="assets/Charts/dom_drivers.png" alt="Dominant Drivers">
            </div>

            <h3>3. Race Characteristics: Lap Times and Evolution</h3>
            <p><strong>Objective:</strong> Examine the evolution of average lap times in races.</p>
            <p><strong>Visualization:</strong> Time series of the average lap time over each year shows the trend of speed in the sport.<br><br> A running average of lap times shows improvements in race speeds over time, influenced by technological advancements and changes in regulations.<br>However, we see lap times in the 1960s became slower due to regulatory changes, including reduced engine capacities and an increased focus on safety.</p>
            <div class="charts">
                <img src="assets/Charts/lap_avg.png" alt="Average Lap Times">
            </div>

            <h3>Conclusion</h3>
            <p>Scuderia Ferrari is clearly the most decorated team historically, but we need to reverse the narrative of recent years.</p>
            <div class="charts">
                <img src="assets/Charts/tifosi.jpg" alt="Tifosi">
            </div>

            <h3>Future Work</h3>
            <ul>
                <li><strong>Deep Dive into Specific Eras:</strong> More detailed analysis of specific periods or drivers.</li>
                <li><strong>Comparison of Team Strategies:</strong> Analyzing pit stop strategies, tyre choices, and other tactical decisions.</li>
                <li><strong>Impact of Technological Changes:</strong> Study how specific technological innovations impacted race outcomes.</li>
            </ul>`,
    link: "https://github.com/robbyhooker/F1-Analysis/blob/main/README.md",
  },
  post4: {
    id: 4,
    title: "Macro Indicator Dashboard",
    synopsis:
      "Live interactive dahsboard with time series plots of macroecnomic indicators",
    content: "",
    link: "",
  },
  post5: {
    id: 5,
    title: "Temporal Trends in Earth's temperature",
    synopsis:
      "Reporting on my the analysis of time series data of Earth's temperature from 1750 to 2015",
    content: `
<p><strong>Author:</strong> Robert Hooker</p>

<h2>Abstract</h2>
<p>This paper reports on the analysis of time series data of Earth's temperature from 1750 to 2015. The goal of this paper was to identify high level climate trends at a global level, as well as changes throughout countries and major cities. The research consists of statistical analysis, visual analyses, and the implementation of an auto-regressive integrated moving average analysis model. The primary data used for analyses was separated into three sets, average temperature globally, by country, and by major city. Due to uncertainty in older measurements, the data was often trimmed to more recent years for reliability purposes. The analyses reveals trends in the global climate, specifically in the past 50 years, that indicate an acceleration in Earth's temperature rise in contrast to the preceding half century. This acceleration is cause for concern for people conscious of the environment. It has already affected the behavior patterns of species that have evolved over millions of years to survive in their natural habitats, and many believe it will be the downfall of our own species. Further research on the subject might consist of utilizing added climate metrics, implementing an auto-regressive distributed lag model, and enhancing visualizations to invigorate a wider audience.</p>

<h2>Keywords</h2>
<p>Global Climate Change, Time Series Analysis, Python Visualizations, Econometrics</p>

<h2>Introduction</h2>
<p>The Earth's temperature has become one of the most widely discussed topics in modern society. The effort to reduce the increase in global temperature often seems futile, but activists hope that further increasing awareness and the amount inescapable data will convince the masses that change is imperative. This paper builds upon previous econometric analyses of the global climate, of which there are many.</p>

<p>Inspiration for this research has been drawn from the work of brilliant climate researchers, who have published extensive works on the factors contributing to, and the trends of the climate change discussed in this paper. Hopefully, this paper serves as an accessible and digestible resource for individuals curious about the data behind climate change. It is designed to act as a gateway, providing a comprehensive yet easily understandable overview of the subject matter. It attempts also to serve as a supplement, and ideally guides people into further reading more extensive climate change research.</p>

<p>The data analyses conducted in this paper were carried out to identify trends in global temperature that signify an accelerated increase in our planet's temperature. Simultaneously, the data visualizations in this paper are meant to increase readability, in an effort to broaden scope and reach. Data science and analysis was carried out in Python, using libraries such as pandas, matplotlib, pmdarima, folium, and others.</p>

<p>Following this introduction, this paper provides a literature review, discussing previous research on the topic. Next, a description of the data, including source, intricacies, supplemental data, and structure. Following this the paper will go into statistical analyses of the important features across the core data sets. Finally, the results of the implemented regression analysis will be discussed, and analyzed. Upon reading these interwoven sections, the reader will have a better high level understanding of the often polarizing subject matter.</p>

<h2>Literature Review</h2>
<p>This literature review aims to provide background information on the topic of global temperature analyses. The topic is widely studied by researchers across disciplines. The papers selected to be reviewed in this section all have in common the fact that they advanced the study of climate change, support their arguments with statistics, and inspired the writing of this paper. This review will detail two papers sequentially, and outline the commonality of their results at the end.</p>

<p>The first paper at hand is David I. Stern and Robert K. Kaufman's 1999 Publishing: <em>Econometric analysis of global climate change</em>. This paper uses rigorous statistical methods to develop an understanding of the causes of global temperature increase, mainly focusing on stochastic trends. The pair conduct a Granger causality test to determine whether or not northern hemisphere temperatures are useful in predicting southern hemisphere temperatures. The paper details the underpinnings of such a test, especially the room for omitted variable bias. Upon conducting the test with 'simple models' with no conditioning variables, the result is a rejection of the null hypothesis, meaning there is south to north causality. However, as they detail in the paper, when they include greenhouse gases in the model, the result is a failed rejection of the null hypothesis, which is a great example of omitted variable bias and how it can skew the results of research. This is admirable as their research was clearly deliberate and carried out by two people who have a profound understanding of econometrics. Next, they test for stochastic trends, which they explain can serve as evidence for causal relation between time series, more so than linear deterministic trends. Upon the carrying out of multiple types of stochastic tests, they explain that results differ due to the different nature of the processes, however the absence of certain integration throughout all tests is helpful to their understanding of the causality. This is an important lesson in statistics, as sometimes it is the non-presence of certain statistics that helps paint the picture.</p>

<p>Moving on to the second paper to be discussed in this review, Camille Parmesan and Gary Yohe published their paper, <em>A globally coherent fingerprint of climate change impacts across natural systems</em> in 2003. Their research focuses on the impact that global climate change has on the species that inhabit our planet, which was a key inspiration for my interest in the broader topic of global warming. The observation that species which have magnificently evolved to survive in their respective domains, have been forced to change their behavior due to the temperature change is a melancholy subject. The authors of this paper meta-analyze over 1,700 species and show biological trends that match those of climate change predictions. Their means of analysis were through species range changes and phonological shifts. Their range analysis shows that on average the range limits of species have moved 6.1 km per decade northward. They also explain that 434 species were changing ranges over the time periods of 17-1,000 years, with a median of 66 years. They add that of these 434, 80% have shifted with climate change predictions. Species are moving to where the earth was previously cooler, and arctic species are contracting their range as the outer bounds of their previous range's heat up. Additionally, their phenological meta-analysis provides similar results. They explain that data from 172 species shows a mean shift towards earlier spring timing of 2.3 days per decade. This result is a simple, yet convincing evidence that the planet is warming, as it shows that our warm seasons are becoming longer, and vice versa. The paper also provides confidence intervals and p-values that show their findings are significant. Similar to the previously discussed paper, the authors are deliberate in their statistical analysis in order to prove its distinction.</p>

<p>From our discussion of just two papers in the field, it is clear that significant thought and resources have gone into the research of climate change. It is also evident that there is a significant and ongoing shift in our planet's climate, captured well by these two statistical studies in temperatures/greenhouse gases, and natural systems behavior. I find it inspiring that these authors dedicate themselves to such a noble cause, especially considering they are well aware of how hard it is to create change in today's society.</p>

<h2>Data Description</h2>
<p>The data used in the analysis for this paper is separated into three sets, all sourced from <a href="https://www.berkeleyearth.org" target="_blank">www.berkeleyearth.org</a>. The most important of these sets perhaps is the Global Temperatures set, as it would be used for the time series regression analysis. The data consists of monthly measurements, and the uncertainty of each measurement from 1750 to 2015. The data also has other features such as max and min temperature, but these would be dropped prior to analysis. It is important to note that all the data used in this analysis would be manipulated and transformed multiple times, depending on the plot or model which it needed to service. For example, this monthly temperature data was transformed into a yearly average for certain plots and for the ARIMA model. A snippet of this important table is shown in figure 1 below.</p>

<figure>
  <img src="assets/climate/global_temp.png" alt="Global Temperature Data">
  <figcaption>Figure 1: Global Temperature Data</figcaption>
</figure>

<p>The second most important data set used in the research for this paper contained monthly measures of temperature for a list of 100 major cities. The data ranged from 1849 to 2013, and had features of average temperature, measurement uncertainty, city, country, and latitude and longitude coordinates. With a large data set such as this one, it is important to ensure cleanliness. To do so, NaN (not a number) values and outliers were removed. In figure 2 below you can see the simple python calculations to remove outliers. Another part of the cleaning process for this set involves the latitude and longitude columns. These columns were original formatted as a general object, with the number coordinate followed by a cardinal direction. This format was undesirable for multiple types of analysis so it would have to be changed. Figure 3 below shows the python function used to transfer to a more usable form, which involves removing the cardinal direction, and instead changing the sign to negative if the coordinate was south or west. This function would be applied to all latitude and longitude values in the major city dataset.</p>

<figure>
  <img src="assets/climate/outliers.png" alt="Outlier Removal">
  <figcaption>Figure 2: Outlier Removal</figcaption>
</figure>

<figure>
  <img src="assets/climate/convertlatlong.png" alt="Latitude and Longitude Conversion">
  <figcaption>Figure 3: Latitude and Longitude Conversion</figcaption>
</figure>

<p>This city data was meant to be used to analyze trends across the globe, which it ended up being very useful for. To do so the data would be grouped into yearly averages by city, while maintaining the latitude and longitude of each city. In python, this means using the 'groupby' method to and taking the mean of each group.</p>

<p>The final dataset using in the research was monthly temperatures by country. Similar to the city data, this data would prove useful in visualizing trends and the fluctuation of said trends across the globe.</p>

<p>In each set the date column is given as a general object. This is not ideal for the methods of analysis used in this research, so the column was transformed into a pandas date-time object. This allows the column to be parsed for year and month individually, and most python statistics packages read date-time objects. Most significant to this paper, the auto arima function from pmdarima, which would be used to perform autoregression, uses date-time objects.</p>

<p>Part of the data cleaning process for all three sets included a look into the uncertainty of measurements column, as using reliable measurements is important when planning to draw conclusions. Upon plotting the average temperature uncertainty by year for the dataset, it was evident that the data should be trimmed down to a period of time where it would be more reliable. Figure 4 below shows the plot used.</p>

<figure>
  <img src="assets/climate/uncert.png" alt="Temperature Uncertainty">
  <figcaption>Figure 4: Temperature Uncertainty</figcaption>
</figure>

<p>From this plot you can see that around the year 1880 the uncertainty of measurements dips below 1° Celsius, which was deemed to be a good spot to trim the data from. It is important to note that when trimming data like this, it is crucial to find a balance between not removing enough bad data that may make your analysis unreliable, and removing too much data, which could make your analysis non-representative of the data.</p>

<p>Going forward, it would be useful to have data that includes more climate metrics than the ones available in these datasets, CO2 emissions for example. Additional features could allow for different types of analysis, such as multiple linear regression. That being said, it is important that this data is from a reliable source in <a href="https://www.berkeleyearth.org" target="_blank">www.berkeleyearth.org</a></p>

<h2>Exploratory Data Analysis</h2>
<p>Once the data was cleaned and sorted, it could be used to understand patterns and trends in the data! These insights are largely the purpose of this paper, as they will hopefully portray climate change data to readers in a digestible form. As discussed in the data description section, the data at hand lacks a large number of variables, however the volume of the data, along with manipulation, allow numerous possibilities for analysis. Prior to observing the analysis, it is important to fully understand the variables at hand. The two variables that appear across each dataset are temperature and temperature uncertainty. These will be used in multiple ways to make up for the lack of other features in the data. It is important to note that temperature variables are measured in Celsius, and any statistical analysis on these variables will be in terms of Celsius.</p>

<p>Firstly, to better understand the temperature, and how poor the measurements were in the early years of this data, we can plot the two metrics together. Using the modified dataframe which averages the yearly temperatures and uncertainty, we can produce a time series plot that tells the story of uncertainty over time. This plot (Figure 5) is also a preview of what is to come in regards to trends in temperature over time.</p>

<figure>
  <img src="assets/climate/temp_uncertainty_fill.png" alt="Temperature and Uncertainty">
  <figcaption>Figure 5: Temperature and Uncertainty</figcaption>
</figure>

<p>In the graph of figure 5, the dark blue line represents the average temperature measurement by year in the global data, where the teal color fill shows the range of uncertainty for each point. It is clear from this visualization why we decided to trim off records of earlier measurements, as they are clearly unreliable. It also appears that in roughly the last 50 years, the rate at which temperature is rising is increasing.</p>

<p>Next, in preparation for time series regression, we will explore time series statistics that will explain some of the data. Specifically we perform a Dickey-Fuller test (one of the tests used in <em>Econometric analysis of global climate change</em>), and check auto correlations of 1, 3, and 5 years. Below is the equation for the Dickey-Fuller test:</p>

<p><code>Δy<sub>t</sub> = δ y<sub>t-1</sub> + u<sub>t</sub></code></p>

<p>Fortunately, we do not have to code this formula into python, as there is an existing function in the stats models package that does this for us. Upon running the function with our yearly average data, we should have statistics that tell us whether or not the data has stationarity.</p>

<table>
  <caption>Dickey Fuller Results</caption>
  <thead>
    <tr>
      <th>Metric</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Test Statistic</td>
      <td>-0.191872</td>
    </tr>
    <tr>
      <td>p-value</td>
      <td>0.939484</td>
    </tr>
    <tr>
      <td>No. of lags used</td>
      <td>15</td>
    </tr>
    <tr>
      <td>Number of observations used</td>
      <td>250</td>
    </tr>
    <tr>
      <td>Critical value (1%)</td>
      <td>-3.456781</td>
    </tr>
    <tr>
      <td>Critical value (5%)</td>
      <td>-2.873172</td>
    </tr>
    <tr>
      <td>Critical value (10%)</td>
      <td>-2.572969</td>
    </tr>
  </tbody>
</table>

<p>From this table we conclude that that the data is non-stationary, as the test statistic is less than smaller than the critical values. This was to be expected based off the graphical representation of the data, and is accounted for in the regression analysis.</p>

<p>Next the auto-correlations of the data will provide us with a measure of the relationship between the current value of temperature and its past values. Table 2 shows the auto-correlation for 1, 3, and 5 year lags.</p>

<table>
  <caption>Autocorrelations for Different Lags</caption>
  <thead>
    <tr>
      <th>Number of lags</th>
      <th>Autocorrelation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>0.72507</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.64145</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.58700</td>
    </tr>
  </tbody>
</table>

<p>We see that as the number of lags increases, the current value becomes less correlated with the lagged values, which is typical in this type of analysis, and will be something to keep in mind during regression.</p>

<p>Next we'll use seasonal decomposition to further identify trends in the data. First simply plugging the yearly average data into a python seasonal decomposition function shows us the trend of the data. This is displayed in figure 6.</p>

<figure>
  <img src="assets/climate/trending.png" alt="Trend Analysis">
  <figcaption>Figure 6: Trend Analysis</figcaption>
</figure>

<p>This chart further solidifies our recognition the rate at which temperature is increasing is accelerating.</p>

<p>Next, using the same seasonal decomposition function, we can analyze trends within a year of data, by inputting monthly temperature data. These results show that global temperature fluctuates throughout the year, and can be seen in figure 7.</p>

<figure>
  <img src="assets/climate/seasonal.png" alt="Seasonal Analysis">
  <figcaption>Figure 7: Seasonal Analysis</figcaption>
</figure>

<p>Moving on, we'd like to identify differences in temperature change across the globe, rather than as a whole. Using the major city and country data, we can map temperatures differences to a map of Earth to help build some intuition behind this.</p>

<p>Both maps will be created using geospatial data and the pandas folium package. The data displayed is a measurement of the difference between a six year average from 2008-2013 and 1900-1905 for each city and country. This measure tells which cities and countries have experienced higher or lower temperature change. Figures 8 and 9 display the city and country map respectively.</p>

<figure>
  <img src="assets/climate/city_map.png" alt="City Temperature Change">
  <figcaption>Figure 8: City Temperature Change</figcaption>
</figure>

<figure>
  <img src="assets/climate/countrymap.png" alt="Country Temperature Change">
  <figcaption>Figure 9: Country Temperature Change</figcaption>
</figure>

<p>The maps are a great indicator of how temperature change varies across the globe, and they also add an element of beauty to the research.</p>

<p>Now, with a better understanding of the data, we can perform regression analysis on both the global and more granular datasets.</p>

<h2>Regression Analysis</h2>
<p>Regression analysis of our temperature data will allow us to try to predict future values, quantify the trends identified in our data, and make a final conclusion on the acceleration of climate change.</p>

<p>Firstly, to analyze trends throughout some of the major cities we can plot each city's average temperature by year, and find a simple linear regression equation to fit the data. The coefficient on the year variable will indicate the magnitude at which temperature is changing in a given city. The charts in figure 10 below provide an example of this from New York City and Santiago. We see that based off the coefficients of the regression lines, New York City's temperature is increasing at a faster rate than Santiago's.</p>

<figure>
  <img src="assets/climate/simplelin.png" alt="City Temperature Trends">
  <figcaption>Figure 10: City Temperature Trends</figcaption>
</figure>

<p>Further graphs like these can be found in the GitHub repository for this project. The graphs are perhaps most useful when used with the city heat map provided in the exploratory analysis section of this paper. Doing so you can see that graphs with greater coefficients correspond to cities that have higher temperature deltas in the map.</p>

<p>Next we will create an autoregressive moving average model using pmdarima's auto arima function. This type of model is ideal for our non-stationary data, as it will transform it to stationary, and then implement autoregressive methods. First, the data we will be using is the global data by yearly average. Due to the uncertainty measures observed earlier in this paper, we have trimmed to data to the range 1900 - 2015 for reliability purposes.</p>

<p>Once the data is in this form, it needs to be split into separate train and test split datasets, where the train dataset is used to create the regression, and the test is compared against predictions from the regression to test the models accuracy. Figure 11 below shows the initial ARIMA model's forecast, as well as the train test split used to create the model.</p>

<figure>
  <img src="assets/climate/ttestplit.png" alt="Train Test Split">
  <figcaption>Figure 11: Train Test Split</figcaption>
</figure>

<p>As you can see, the model is an awful predictor of future values or global temperature. The root mean squared error came in at .3167, which is huge in terms of Celsius measurement. Upon looking at the train test split and the trend of our forecast, it seems the train portion of the model is a poor range to use for training, as it does not represent the more recent changes in global climate.</p>

<p>In an effort to make the model a better predictor, we can move the train and test split up to the portion of the data where we start to see similar trends to those of current values. In figure 12, the new train test split is shown, where it begins in 1970. Also plotted is the new model's predictions in blue, and the old model's predictions in red.</p>

<figure>
  <img src="assets/climate/newttestsplit.png" alt="New Train Test Split">
  <figcaption>Figure 12: New Train Test Split</figcaption>
</figure>

<p>Although trimming the data, reduces our volume of predictions, based on the graph, it is clearly a better predictor of future temperature values. This is also evidenced by the new, lower, root mean squared error of .1251. Table 3 below contains statistics describing the newly generated model.</p>

<table>
  <caption>SARIMAX Model Results</caption>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Coefficient</th>
      <th>Standard Error</th>
      <th>Z-Score</th>
      <th>P-Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Intercept</td>
      <td>0.0652</td>
      <td>0.041</td>
      <td>1.604</td>
      <td>0.109</td>
    </tr>
    <tr>
      <td>ar.L1</td>
      <td>-0.8058</td>
      <td>0.201</td>
      <td>-4.001</td>
      <td>0.000</td>
    </tr>
    <tr>
      <td>ar.L2</td>
      <td>-0.5363</td>
      <td>0.227</td>
      <td>-2.359</td>
      <td>0.018</td>
    </tr>
    <tr>
      <td>ar.L3</td>
      <td>-0.2939</td>
      <td>0.180</td>
      <td>-1.631</td>
      <td>0.103</td>
    </tr>
    <tr>
      <td>σ^2</td>
      <td>0.0425</td>
      <td>0.014</td>
      <td>2.963</td>
      <td>0.003</td>
    </tr>
  </tbody>
</table>

<p>We can see that the coefficients on values at t-1 and t-2 are significant, as their P-Values are less than .05, however the coefficient on values at t-3 are not significant. A quick interpretation of the summary statistics provides insight to the performance and reliability of the model. The high Log Likelihood shows that the model explains the observed data well, and the low AIC suggests the same. The low P-Value for Ljung-Box (L1)(Q) suggests that there is autocorrelation in the data, which was also observed earlier in this paper.</p>

<h2>Conclusion</h2>
<p>To wrap up this paper we can conclude that all data used in this analysis of global temperatures points to an accelerated increase in the Earth's temperature. It also helps us develop intuition on how the temperature change varies across the globe. The paper builds off econometric style research, and reports similar findings to the widely accepted narrative related to this subject. The data used to perform this analysis was simple, but very versatile, and being sourced from <a href="https://www.berkeleyearth.org" target="_blank">www.berkeleyearth.org</a>, it can be considered reliable. Upon statistical analysis of the variables in the data we see graphs and maps that indicate global temperature increase is accelerating. The same conclusion can be drawn from the regression analysis carried out for this paper, specifically showing that temperatures in roughly the last 50 years are increasing more rapidly than the preceding half-century. Hopefully after reading through the sections of this paper, you have gained an understanding of the trends in Earth's temperature. Furthermore, hopefully more research like this acts as a catalyst for change, so we as a species can prevent further damage to our planet's natural systems.</p>

<h2>References</h2>
<ul>
  <li>David I. Stern, Robert K. Kaufmann, Econometric analysis of global climate change, Environmental Modelling & Software, Volume 14, Issue 6, 1999, Pages 597-605, ISSN 1364-8152, <a href="https://doi.org/10.1016/S1364-8152(98)00094-2">https://doi.org/10.1016/S1364-8152(98)00094-2</a>. (<a href="https://www.sciencedirect.com/science/article/pii/S1364815298000942">https://www.sciencedirect.com/science/article/pii/S1364815298000942</a>)</li>
  <li>Parmesan, C., & Yohe, G. (2003). A globally coherent fingerprint of climate change impacts across natural systems. Integrative Biology, Patterson Laboratories 141, University of Texas, Austin, Texas 78712, USA. John E. Andrus Professor of Economics, Wesleyan University, 238 Public Affairs Center, Middletown, Connecticut 06459, USA.</li>
  <li>Pierre, S. (2022). A Guide to Time Series Analysis in Python. <a href="https://builtin.com/data-science/time-series-python">https://builtin.com/data-science/time-series-python</a></li>
</ul>
`,
    link: "https://github.com/robbyhooker/Earth-Temperature-Analysis/tree/main",
  },
  post6: {
    id: 6,
    title: "Credit Risk Modeling",
    synopsis:
      "This project aims to develop a statistical model to predict loan defaults based on borrower information",
    content: `
        <h2>Project Overview</h2>
        <p>This project aims to develop a statistical model to predict loan defaults based on borrower information. In deployment, the model could help financial institutions assess the risk associated with lending to various applicants, thereby reducing potential financial losses.</p>

        <h2>Table of Contents</h2>
        <ul>
            <li><a href="#dataset-description">Dataset Description</a></li>
            <li><a href="#data-preprocessing">Data Preprocessing</a></li>
            <li><a href="#feature-engineering">Feature Engineering</a></li>
            <li><a href="#model-building">Model Building</a></li>
            <li><a href="#model-evaluation">Model Evaluation</a></li>
            <li><a href="#results-and-insights">Results and Insights</a></li>
            <li><a href="#request-example">Request Example</a></li>
            <li><a href="#future-work">Future Work</a></li>
        </ul>

        <h2 id="dataset-description">Dataset Description</h2>
        <p>The dataset used in this project is sourced from the <a href="https://www.kaggle.com/datasets/laotse/credit-risk-dataset" target="_blank">Kaggle</a>. It includes features such as age, income, employment length, loan amount, loan interest rate, and others. The target variable is whether the borrower defaulted on the loan.</p>

        <h3>Features</h3>
        <ul>
            <li><code>person_age</code>: Age of the borrower</li>
            <li><code>person_income</code>: Annual income of the borrower</li>
            <li><code>person_emp_length</code>: Length of employment in years</li>
            <li><code>loan_amnt</code>: Loan amount requested</li>
            <li><code>loan_int_rate</code>: Interest rate on the loan</li>
            <li><code>loan_percent_income</code>: Percentage of income that goes towards loan payments</li>
            <li><code>cb_person_cred_hist_length</code>: Length of credit history in years</li>
            <li><code>person_home_ownership</code>: Home ownership status (Own, Rent, Mortgage, Other)</li>
            <li><code>loan_intent</code>: Purpose of the loan (Education, Home Improvement, Medical, Personal, Venture)</li>
        </ul>

        <h2 id="data-preprocessing">Data Preprocessing</h2>
        <h3>Steps Taken</h3>
        <ul>
            <li><strong>Handling Missing Values</strong>: Missing values were imputed using median or mode values.</li>
            <li><strong>Outlier Removal</strong>: Outliers in numerical features were removed using the IQR method.</li>
            <li><strong>Encoding Categorical Variables</strong>: Categorical variables were encoded using one-hot encoding.</li>
        </ul>

        <h2>Exploratory Data Analysis</h2>
        <h3>Distribution of Key Variables</h3>
        <p>The distribution of key variables such as <code>person_age</code>, <code>person_income</code>, <code>loan_amnt</code>, and <code>loan_int_rate</code> can provide insights into the characteristics of the borrowers.</p>

        <h4>Distribution of Person Age</h4>
        <p>We see from the below distribution that borrowers in this dataset tend to be younger. The intuition behind this is that typically older people have built up an amount of wealth that prevents them from needing to borrow.</p>
        <img src="assets/Charts/age_distribution.png" alt="Distribution of Person Age">

        <h4>Distribution of Person Income</h4>
        <p>We also see that majority of the persons income in this dataset fall within $30,000 - $80,000. Note this number is not representative of the population, rather the incomes of people who are taking loans.</p>
        <img src="assets/Charts/income_distribution.png" alt="Distribution of Person Income">

        <h4>Distribution of Loan Amount</h4>
        <p>Considering the majority of incomes in the dataset, we would expect loan amount distribution to skew left. Indeed, the loans are mostly under $20,000.</p>
        <img src="assets/Charts/loan_distribution.png" alt="Distribution of Loan Amount">

        <h4>Distribution of Loan Interest Rate</h4>
        <p>Without credit scores in the dataset the interest rates on the loans could be useful in gauging applicants, as typically better credit scores are rewarded with relatively lower interest rates. In this case the rates are somewhat evenly distributed, and this feature may be more useful at the individual level.</p>
        <img src="assets/Charts/rate_distribution.png" alt="Distribution of Loan Interest Rate">

        <h3>Scatter Plots of Variables and Defaults</h3>
        <p>Scatter plots with default indicators can help us visualize and identify groups who are at higher risk of default.</p>

        <h4>Person Income vs Loan Interest Rate (1 = Default)</h4>
        <p>This chart is a visualization of how high interest and low income is a recipe for disaster. Another interesting takeaway is that high rates trend towards default for all income levels, which could be an indicator of people with poor credit history (hence the high interest rate) sustaining their habit.</p>
        <img src="assets/Charts/income_interestrate.png" alt="Person Income vs Loan Interest Rate">

        <h4>Interest Rate vs Loan Amount</h4>
        <p>Also confirming the high interest rate-high default rate trend. This chart however does not show any obvious correlation between loan amount and default likelihood.</p>
        <img src="assets/Charts/interestrate_loanamnt.png" alt="Interest Rate vs Loan Amount">

        <h4>Person Income vs Loan Amount</h4>
        <p>This chart seems to point out the obvious but it is slightly jarring to look at. People taking loans that amount to a relatively large portion of their income are almost guaranteed to default. This group is the red diagonal on the left side of the plot.</p>
        <img src="assets/Charts/income_loanamnt.png" alt="Person Income vs Loan Amount">

        <h3>Default Rates by Categories</h3>
        <p>Default rates by categories such as <code>home_ownership</code>, <code>loan_intent</code>, and <code>loan_grade</code> provide insights into which groups are more likely to default.</p>

        <h4>Default Rates by Home Ownership</h4>
        <p>This chart indicates that a person who does not own their home (or working towards owning it) is more than twice as likely to default on a loan as someone who does. This is unsurprising but another discouraging piece of information for renters who cannot afford to buy a home in the current economy.</p>
        <img src="assets/Charts/dr_home.png" alt="Default Rates Home Ownership">

        <h4>Default Rates by Loan Grade</h4>
        <p>Portraying the obvious here... but it is staggering that over 98% of G grade loan in this data set ended in defaults.</p>
        <img src="assets/Charts/dr_grade.png" alt="Default Rates by Loan Grade">

        <h4>Default Rates by Loan Intent</h4>
        <p>Not a ton of trends in this chart, although it is nice to see education with relatively low default rates. This could be due to the longer time to amortize, or perhaps educated people better understand loans and the dangers of them. What stands out the most is that debt consolidation has the highest default rate... this feels like a bummer.</p>
        <img src="assets/Charts/dr_intent.png" alt="Default Rates by Loan Intent">

        <h2 id="model-building">Model Building</h2>
        <p>Multiple models were built and compared using sklearn's modeling and metric packages. The models tested in this project include:</p>
        <ul>
            <li>Logistic Regression</li>
            <li>Decision Tree</li>
            <li>Random Forest</li>
        </ul>
        <p>Hyperparameter tuning was performed using GridSearchCV to find the best parameters for the random forest model.</p>

        <h2 id="model-evaluation">Model Evaluation</h2>
        <p>Models were evaluated using various metrics:</p>
        <ul>
            <li><strong>Confusion Matrix</strong>: To visualize the performance in terms of true positives, true negatives, false positives, and false negatives.</li>
            <li><strong>ROC Curve</strong>: To evaluate the model's ability to distinguish between classes (Default vs Non-Default).</li>
            <li><strong>AUC Score</strong>: Area under the ROC Curve to quantify the overall performance.</li>
        </ul>

        <h3>Model Performance</h3>
        <ul>
            <li><strong>Logistic Regression</strong>: ROC AUC = 0.85</li>
            <li><strong>Decision Tree</strong>: ROC AUC = 0.87</li>
            <li><strong>Random Forest</strong>: ROC AUC = 0.94 (Best Performing Model)</li>
        </ul>

        <h2 id="results-and-insights">Results and Insights</h2>
        <p>The Random Forest model showed the best performance with an ROC AUC score of 0.94. Below we summarize the winning model:</p>

        <h3>Feature Importances</h3>
        <p>The feature importances in the Random Forest model show which variables are most predictive of loan default. The most important features are <code>loan_percent_income</code>, <code>person_income</code>, <code>loan_int_rate</code>, and <code>loan_grade</code>. These are all intuitively important, but it is nice to verify that the model recognizes this. It is also interesting to note that age and credit length aren't a very big factor in predicting default. This somewhat suggests that being a reckless debtor is inherent, and not something that can be easily learned away. <strong>However, this is a large claim and would take enormous amounts of social research to support!</strong></p>
        <img src="assets/Charts/feature_importance.png" alt="Feature Importances">

        <h3>Model Performance Metrics</h3>
        <p>The performance of the Random Forest model is evaluated using ROC Curve and Confusion Matrix.</p>

        <h4>ROC Curve</h4>
        <p><strong>Reading the ROC Curve</strong>: Allowing for 0 false positives (predicting default when the case is non-default) we obtain some rate of true positives, in this case that is somewhere between 0.6 - 0.8. Now as we allow for more false positives the model also has a higher true positive rate and the curve goes up. The area under the curve (AUC) score is a common metric for model performance, and the ideal predictor is a horizontal line at y = 1.0.</p>
        <img src="assets/Charts/rf_auc.png" alt="ROC Curve">

        <h4>Confusion Matrix</h4>
        <img src="assets/Charts/rf_conf.png" alt="Confusion Matrix">

        <h2 id="request-example">Request Example</h2>
        <p>Below we give the model a (made up) person's loan information, and the model returns an adequate prediction.</p>

        <pre><code>{
  "person_age": 45,
  "person_income": 55000,
  "person_emp_length": 2,
  "loan_amnt": 15000,
  "loan_int_rate": 11.25,
  "loan_percent_income": 0.170203,
  "cb_person_cred_hist_length": 3,
  "person_home_ownership_OTHER": 0, 
  "person_home_ownership_OWN": 1,
  "person_home_ownership_RENT": 0, 
  "loan_intent_EDUCATION": 1,
  "loan_intent_HOMEIMPROVEMENT": 0, 
  "loan_intent_MEDICAL": 0,
  "loan_intent_PERSONAL": 0, 
  "loan_intent_VENTURE": 0,
  "loan_grade_B": 1,
  "loan_grade_C": 0,
  "loan_grade_D": 0,
  "loan_grade_E": 0,
  "loan_grade_F": 0,
  "loan_grade_G": 0,
  "cb_person_default_on_file_Y": 0
}
</code></pre>
        <p>Response:<br>
        Model Default Prediction: 0 (non-default)<br> 
        Model Confidence in Prediction: 0.95%</p>

        <h2 id="future-work">Future Work</h2>
        <p>Potential future improvements include:</p>
        <ul>
            <li>Incorporating additional features such as credit score.</li>
            <li>Get way more data and build a neural network :)</li>
            <li>Implementing a more sophisticated handling of missing data.</li>
            <li>Productionize the model into an API and allow public use.</li>
        </ul>
        <h2>Thanks for reading!!</h2>
        <p>I love the idea of machines assisiting in managing risk, but it is scary to invision them having full decision making responsibilty. Biases can be baked into models in subtle ways and certain groups could get the short end of this stick and be unfairly treated. Hence it is best to always involve human intuition for human decisions. Anyways, I always enjoy working with data like this as it is fun to speculate on and helps me better understand our world, one niche topic at a time:)
</p>
        `,
    link: "https://github.com/robbyhooker/Credit-Risk-Modeling",
  },
  post7: {
    id: 7,
    title: "Mock E-Commerce Dashboard",
    synopsis:
      "Live interactive Tableau dahsboard with different metrics from e-commerce sales data",
    content: "",
    link: "",
  },
};

document.addEventListener("DOMContentLoaded", () => {
  const urlParams = new URLSearchParams(window.location.search);
  const postId = urlParams.get("id");
  const post = projects[postId];

  if (post) {
    document.getElementById("project-post").innerHTML = `
            <title>${post.title}</title>
            <h1>${post.title}</h1>
            <div class="content">${post.content}</div>
            <a href="${post.link}" class="more-button" target="_blank">Github</a>
        `;
  } else {
    document.getElementById(
      "project-post"
    ).innerHTML = `<h1 class="notfound">Post not found</h1>`;
  }
});
