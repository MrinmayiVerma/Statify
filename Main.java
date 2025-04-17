package org.statlib.core;

import java.util.ArrayList;
import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        ArrayList<Double> randomNumbers = StatLibrary.generateRandomNumbers();
        double[] values = randomNumbers.stream().mapToDouble(Double::doubleValue).toArray();

        double avgDeviation = StatLibrary.computeAverageDeviation(values);
        System.out.println("Average Deviation: " + avgDeviation);

        double average = StatLibrary.computeAverage(values);
        System.out.println("Average: " + average);

        Object[] mixedValues = {10, 20.5, true, false, 15};
        double avgMixed = StatLibrary.computeArguments(mixedValues);
        System.out.println("Average of mixed values: " + avgMixed);

        double avgCells = StatLibrary.computeAverageCells(values, ">", 50);
        System.out.println("Average of values greater than 50: " + avgCells);

        double betaFunctionResult = StatLibrary.betaFunction(2, 3);
        System.out.println("Beta Function(2,3): " + betaFunctionResult);

        double betaDistResult = StatLibrary.computeBetaDist(0.5, 2, 3);
        System.out.println("Beta Distribution(0.5, 2, 3): " + betaDistResult);

        double betaInv = StatLibrary.computeBetaInv(0.6, 2, 3);
        System.out.println("Beta Inverse(0.6, 2, 3): " + betaInv);

        double binomDist = StatLibrary.computeBinomDist(2, 5, 0.5);
        System.out.println("Binomial Distribution(2, 5, 0.5): " + binomDist);

        double binomDistRange = StatLibrary.computeBinomDistRange(5, 0.5, 2, 4);
        System.out.println("Binomial Distribution Range(5, 0.5, 2-4): " + binomDistRange);

        int binomInv = StatLibrary.computeBinomInv(5, 0.5, 0.8);
        System.out.println("Binomial Inverse(5, 0.5, 0.8): " + binomInv);

        double chiSqCDF = StatLibrary.chiSqCDF(3.84, 1);
        System.out.println("Chi-Square CDF(3.84, 1): " + chiSqCDF);

        double chiSqRT = StatLibrary.computeChiSqDistRT(3.84, 1);
        System.out.println("Chi-Square Right-Tail(3.84, 1): " + chiSqRT);

        double chiSqInv = StatLibrary.computeChiSqInv(0.95, 1);
        System.out.println("Chi-Square Inverse(0.95, 1): " + chiSqInv);

        double chiSqInvRT = StatLibrary.computeChiSqInvRT(0.05, 1);
        System.out.println("Chi-Square Inverse Right-Tail(0.05, 1): " + chiSqInvRT);

        double[][] observed = {{10, 20}, {30, 40}};
        double[][] expected = {{15, 15}, {25, 45}};
        double chiSqTest = StatLibrary.computeChiSqTest(observed, expected);
        System.out.println("Chi-Square Test: " + chiSqTest);

        double confidenceNorm = StatLibrary.computeConfidenceNorm(0.05, 5, 30);
        System.out.println("Confidence Interval (Normal): " + confidenceNorm);

        double confidenceT = StatLibrary.computeConfidenceT(0.05, 5, 30);
        System.out.println("Confidence Interval (T-distribution): " + confidenceT);

        double[] xValues = {1, 2, 3, 4, 5};
        double[] yValues = {2, 4, 6, 8, 10};

        double correlation = StatLibrary.computeCorrel(xValues, yValues);
        System.out.println("Correlation: " + correlation);

        mixedValues = new Object[]{10, "text", 20.5, true, null, 15};
        int countNumbers = StatLibrary.computeCount(mixedValues);
        System.out.println("Count of numbers: " + countNumbers);

        int countAll = StatLibrary.computeCountA(mixedValues);
        System.out.println("Count of all values: " + countAll);

        String[] textValues = {"hello", "", "world", null, " ", "java"};
        int countBlank = StatLibrary.computeCountBlank(textValues);
        System.out.println("Count of blank values: " + countBlank);

        Integer[] numArray = {5, 10, 15, 20, 25};
        int countIf = StatLibrary.computeCountIf(numArray, n -> n > 10);
        System.out.println("Count of numbers > 10: " + countIf);

        Predicate<Integer>[] conditions = new Predicate[]{
                n -> n > 10,
                n -> n % 2 == 0
        };
        int countIfS = StatLibrary.computeCountIfS(numArray, conditions);
        System.out.println("Count of numbers > 10 and even: " + countIfS);

        double covariance = StatLibrary.computeCovarianceP(xValues, yValues);
        System.out.println("Covariance: " + covariance);

        double[] xValues = {1, 2, 3, 4, 5};
        double[] yValues = {2, 4, 6, 8, 10};

        double covarianceS = StatLibrary.computeCovarianceS(xValues, yValues);
        System.out.println("Sample Covariance: " + covarianceS);

        double devSq = StatLibrary.computeDevSq(xValues);
        System.out.println("Sum of Squares of Deviations: " + devSq);

        double exponDist = StatLibrary.computeExponDist(2, 0.5, true);
        System.out.println("Exponential Distribution: " + exponDist);

        double fDist = StatLibrary.computeFDist(3.5, 5, 10, true);
        System.out.println("F Distribution: " + fDist);

        double fInv = StatLibrary.computeFInv(0.95, 5, 10);
        System.out.println("Inverse F Distribution: " + fInv);

        double[] sample1 = {10, 12, 14, 16, 18};
        double[] sample2 = {20, 22, 24, 26, 28};
        double fTest = StatLibrary.computeFTest(sample1, sample2);
        System.out.println("F-Test Result: " + fTest);

        double fisher = StatLibrary.computeFisher(0.5);
        System.out.println("Fisher Transformation: " + fisher);

        double fisherInv = StatLibrary.computeFisherInv(0.2);
        System.out.println("Inverse Fisher Transformation: " + fisherInv);

        double forecast = StatLibrary.computeForecast(6, xValues, yValues);
        System.out.println("Forecast Value: " + forecast);

        double[] values = {10, 12, 15, 14, 18, 20, 22, 24};
        int period = 3;
        double confidenceLevel = 1.96;

        double forecastETS = StatLibrary.computeForecastETS(values, period);
        System.out.println("Forecast ETS: " + forecastETS);

        double confInt = StatLibrary.computeForecastETSConfInt(values, period, confidenceLevel);
        System.out.println("Confidence Interval: " + confInt);

        int seasonality = StatLibrary.computeForecastETSSeasonality(values);
        System.out.println("Seasonality: " + seasonality);

        double forecastStat = StatLibrary.computeForecastETSStat(values);
        System.out.println("Forecast ETS Statistic: " + forecastStat);

        double[] knownX = {1, 2, 3, 4, 5};
        double[] knownY = {2, 4, 6, 8, 10};
        double linearForecast = StatLibrary.computeForecastLinear(6, knownX, knownY);
        System.out.println("Linear Forecast: " + linearForecast);

        double[] data = {3.5, 7.8, 10.2, 15.6, 18.4, 20.1, 25.3};
        double[] bins = {5, 10, 15, 20};
        int[] frequency = StatLibrary.computeFrequency(data, bins);
        System.out.println("Frequency Distribution: " + Arrays.toString(frequency));

        double x = 5.0;
        double alpha = 2.0;
        double beta = 1.0;
        double probability = 0.5;
        
        System.out.println("Gamma(" + x + ") = " + computeGamma(x));
        System.out.println("Gamma Distribution(" + x + ") = " + computeGammaDist(x, alpha, beta, false));
        System.out.println("Cumulative Gamma Distribution(" + x + ") = " + computeGammaDist(x, alpha, beta, true));
        System.out.println("Inverse Gamma Distribution(" + probability + ") = " + computeGammaInv(probability, alpha, beta));
        System.out.println("Log Gamma(" + x + ") = " + computeGammaLn(x));
        System.out.println("Precise Log Gamma(" + x + ") = " + computeGammaLnPrecise(x));
        System.out.println("Gaussian(" + x + ") = " + computeGauss(x));

        double[] values = {1.0, 3.0, 9.0, 27.0};
        System.out.println("Geometric Mean = " + computeGeomean(values));

        double[] knownY = {2.0, 4.0, 8.0, 16.0};
        double[] knownX = {1.0, 2.0, 3.0, 4.0};
        double[] newX = {5.0, 6.0};
        double[] growthResults = computeGrowth(knownY, knownX, newX);
        
        System.out.print("Growth Results: ");
        for (double result : growthResults) {
            System.out.print(result + " ");
        }
        System.out.println();
        double[] values = {1, 2, 4, 8, 16};
        System.out.println("Harmonic Mean: " + computeHarmean(values));
        
        int x = 2, M = 10, n = 5, N = 3;
        System.out.println("Hypergeometric Distribution: " + computeHypergeomDist(x, M, n, N, false));
        
        double[] xValues = {1, 2, 3, 4, 5};
        double[] yValues = {2, 4, 6, 8, 10};
        System.out.println("Linear Regression Intercept: " + computeIntercept(xValues, yValues));
        
        double[] dataset = {3, 7, 7, 2, 4, 9, 10};
        System.out.println("Kurtosis: " + computeKurtosis(dataset));
        
        int k = 2;
        System.out.println("2nd Largest Value: " + computeLarge(dataset, k));
        
        double[] linestParams = computeLinest(xValues, yValues);
        System.out.println("Linear Trend Slope: " + linestParams[0] + ", Intercept: " + linestParams[1]);

        double[] values = {1, 2, 4, 8, 16};
        System.out.println("Harmonic Mean: " + computeHarmean(values));
        
        int x = 2, M = 10, n = 5, N = 3;
        System.out.println("Hypergeometric Distribution: " + computeHypergeomDist(x, M, n, N, false));
        
        double[] xValues = {1, 2, 3, 4, 5};
        double[] yValues = {2, 4, 6, 8, 10};
        System.out.println("Linear Regression Intercept: " + computeIntercept(xValues, yValues));
        
        double[] dataset = {3, 7, 7, 2, 4, 9, 10};
        System.out.println("Kurtosis: " + computeKurtosis(dataset));
        
        int k = 2;
        System.out.println("2nd Largest Value: " + computeLarge(dataset, k));
        
        double[] linestParams = computeLinest(xValues, yValues);
        System.out.println("Linear Trend Slope: " + linestParams[0] + ", Intercept: " + linestParams[1]);
        
        double[] logestParams = computeLogest(xValues, yValues);
        System.out.println("Exponential Trend Base: " + logestParams[0] + ", Growth Rate: " + logestParams[1]);
        
        System.out.println("Maximum Value: " + computeMax(dataset));
        System.out.println("Minimum Value: " + computeMin(dataset));
        System.out.println("Median: " + computeMedian(dataset));
        double mean = 0;
        double stddev = 1;
        double probability = 0.95;
        double z = 1.645;

        
        System.out.println("Inverse Normal Distribution: " + Statistics.computeNormInv(probability, mean, stddev));
        
        
        System.out.println("Standard Normal CDF: " + Statistics.computeormSDist(z, true));
        
        
        System.out.println("Inverse Standard Normal CDF: " + Statistics.computeNormSInv(probability));
        
        
        double[] x = {1, 2, 3, 4, 5};
        double[] y = {2, 3, 5, 7, 11};
        System.out.println("Pearson Correlation: " + Statistics.computePearson(x, y));
        
        
        double[] values = {1, 2, 3, 4, 5};
        double k = 0.5;
        System.out.println("Percentile Exclusive: " + Statistics.computePercentileExc(values, k));
        
        
        System.out.println("Percentile Inclusive: " + Statistics.computePercentileInc(values, k));
        
        
        double rankValue = 3;
        System.out.println("Percent Rank Inclusive: " + Statistics.computePercentRankInc(values, rankValue));
        
        
        int n = 5, r = 3;
        System.out.println("Permutations: " + Statistics.computePermut(n, r));

        double mean = 0;
        double stddev = 1;
        double probability = 0.95;
        double z = 1.645;

        // Compute Inverse Normal Distribution
        System.out.println("Inverse Normal Distribution: " + Statistics.computeNormInv(probability, mean, stddev));
        
        // Compute Standard Normal CDF
        System.out.println("Standard Normal CDF: " + Statistics.computeormSDist(z, true));
        
        // Compute Inverse of Standard Normal CDF
        System.out.println("Inverse Standard Normal CDF: " + Statistics.computeNormSInv(probability));
        
        // Compute Pearson Correlation Coefficient
        double[] x = {1, 2, 3, 4, 5};
        double[] y = {2, 3, 5, 7, 11};
        System.out.println("Pearson Correlation: " + Statistics.computePearson(x, y));
        
        // Compute Percentile Exclusive
        double[] values = {1, 2, 3, 4, 5};
        double k = 0.5;
        System.out.println("Percentile Exclusive: " + Statistics.computePercentileExc(values, k));
        
        // Compute Percentile Inclusive
        System.out.println("Percentile Inclusive: " + Statistics.computePercentileInc(values, k));
        
        // Compute Percent Rank Inclusive
        double rankValue = 3;
        System.out.println("Percent Rank Inclusive: " + Statistics.computePercentRankInc(values, rankValue));
        
        // Compute Permutations
        int n = 5, r = 3;
        System.out.println("Permutations: " + Statistics.computePermut(n, r));
        
        // Compute Permutations with Repetition
        System.out.println("Permutations with Repetition: " + Statistics.computePermutationA(n, r));
        
        // Compute Standard Normal Density Function
        System.out.println("Standard Normal Density Function: " + Statistics.computePhi(z));
        
        // Compute Poisson Distribution
        int xVal = 3;
        System.out.println("Poisson Distribution: " + Statistics.computePoissonDist(xVal, mean, false));
        
        // Compute Probability within Range
        double lowerLimit = 2, upperLimit = 4;
        System.out.println("Probability within Range: " + Statistics.computeProb(values, lowerLimit, upperLimit));
        
        // Compute Quartile Exclusive
        int quartile = 2;
        System.out.println("Quartile Exclusive: " + Statistics.computeQuartileExc(values, quartile));
        
        // Compute Quartile Inclusive
        System.out.println("Quartile Inclusive: " + Statistics.computeQuartileInc(values, quartile));
        
        // Compute Rank Average
        double rankAvgValue = 3;
        System.out.println("Rank Average: " + Statistics.computeRankAvg(values, rankAvgValue));
        
        // Compute Rank Equal
        System.out.println("Rank Equal: " + Statistics.computeRankEq(values, rankAvgValue));
    // Compute Skewness
    System.out.println("Skewness: " + Statistics.computeSkew(values));
        
    // Compute Population Skewness
    System.out.println("Population Skewness: " + Statistics.computeSkewP(values));
    
    // Compute Slope of Linear Regression
    System.out.println("Slope: " + Statistics.computeSlope(x, y));
    
    // Compute kth Smallest Value
    int kSmall = 2;
    System.out.println("kth Smallest Value: " + Statistics.computeSmall(values, kSmall));
    
    // Compute Standardized Value
    double standardizeValue = 3;
    System.out.println("Standardized Value: " + Statistics.computeStandardize(standardizeValue, mean, stddev));
    
    // Compute Population Standard Deviation
    System.out.println("Population Standard Deviation: " + Statistics.computeStdevP(values));
    
    // Compute Sample Standard Deviation
    System.out.println("Sample Standard Deviation: " + Statistics.computeStdevS(values));
    
    // Compute Standard Error of Regression
    System.out.println("Standard Error of Regression: " + Statistics.computeSteyx(x, y));
    
    // Compute Student t-Distribution
    int degreesOfFreedom = 10;
    System.out.println("Student t-Distribution: " + Statistics.computeTDist(z, degreesOfFreedom));
    
    // Compute Two-Tailed t-Distribution
    System.out.println("Two-Tailed t-Distribution: " + Statistics.computeTDist2T(z, degreesOfFreedom));
    
    // Compute Right-Tailed t-Distribution
    System.out.println("Right-Tailed t-Distribution: " + Statistics.computeTDistRT(z, degreesOfFreedom));
    
    // Compute t-Test
    double[] sample1 = {1.1, 2.3, 3.5, 4.7, 5.9};
    double[] sample2 = {1.2, 2.4, 3.6, 4.8, 6.0};
    System.out.println("t-Test: " + Statistics.computeTTest(sample1, sample2));
    // Compute Inverse Normal Distribution
    System.out.println("Inverse Normal Distribution: " + Statistics.computeNormInv(probability, mean, stddev));
        
    // Compute Standard Normal CDF
    System.out.println("Standard Normal CDF: " + Statistics.computeNormSDist(z, true));
    
    // Compute Inverse of Standard Normal CDF
    System.out.println("Inverse Standard Normal CDF: " + Statistics.computeNormSInv(probability));
    
    // Compute Pearson Correlation Coefficient
    double[] x = {1, 2, 3, 4, 5};
    double[] y = {2, 3, 5, 7, 11};
    System.out.println("Pearson Correlation: " + Statistics.computePearson(x, y));
    
    // Compute Percentile Exclusive
    double[] values = {1, 2, 3, 4, 5};
    double k = 0.5;
    System.out.println("Percentile Exclusive: " + Statistics.computePercentileExc(values, k));
    
    // Compute Percentile Inclusive
    System.out.println("Percentile Inclusive: " + Statistics.computePercentileInc(values, k));
    
    // Compute Standard Deviation
    System.out.println("Population Standard Deviation: " + Statistics.computeStdevP(values));
    System.out.println("Sample Standard Deviation: " + Statistics.computeStdevS(values));
    
    // Compute Variance
    System.out.println("Population Variance: " + Statistics.computeVarP(values));
    System.out.println("Sample Variance: " + Statistics.computeVarS(values));
    
    // Compute Skewness
    System.out.println("Skewness: " + Statistics.computeSkew(values));
    System.out.println("Population Skewness: " + Statistics.computeSkewP(values));
    
    // Compute Slope of Linear Regression
    System.out.println("Slope: " + Statistics.computeSlope(x, y));
    
    // Compute Standard Error of Regression
    System.out.println("Standard Error of Regression: " + Statistics.computeSteyx(x, y));
    
    // Compute Student t-Distribution
    int degreesOfFreedom = 10;
    System.out.println("Student t-Distribution: " + Statistics.computeTDist(z, degreesOfFreedom));
    System.out.println("Two-Tailed t-Distribution: " + Statistics.computeTDist2T(z, degreesOfFreedom));
    System.out.println("Right-Tailed t-Distribution: " + Statistics.computeTDistRT(z, degreesOfFreedom));
    
    // Compute t-Test
    double[] sample1 = {1.1, 2.3, 3.5, 4.7, 5.9};
    double[] sample2 = {1.2, 2.4, 3.6, 4.8, 6.0};
    System.out.println("t-Test: " + Statistics.computeTTest(sample1, sample2));
    
    // Compute Trend
    double[] newX = {6, 7, 8};
    System.out.println("Trend Predictions: " + Arrays.toString(Statistics.computeTrend(x, y, newX)));
    
    // Compute Trimmed Mean
    double trimPercent = 0.2;
    System.out.println("Trimmed Mean: " + Statistics.computeTrimMean(values, trimPercent));
    
    // Compute Smallest kth Value
    int kSmall = 2;
    System.out.println("Smallest " + kSmall + "th Value: " + Statistics.computeSmall(values, kSmall));
    
    // Compute Standardized Value
    double valueToStandardize = 3;
    System.out.println("Standardized Value: " + Statistics.computeStandardize(valueToStandardize, mean, stddev));
    double[] values = {10, 12, 23, 23, 16, 23, 21, 16};
        double[] values2 = {15, 18, 25, 30, 22, 28, 24, 19};
        Object[] mixedValues = {10, "text", 15, true, 20, 25};

        // Standard deviation calculations
        System.out.println("Population Standard Deviation: " + Statistics.computeStdevP(values));
        System.out.println("Sample Standard Deviation: " + Statistics.computeStdevS(values));
        System.out.println("Sample Standard Deviation (Including Non-Numeric Values): " + Statistics.computeStdevA(mixedValues));
        System.out.println("Population Standard Deviation (Including Non-Numeric Values): " + Statistics.computeStdevPA(mixedValues));
        
        // Regression standard error
        System.out.println("Standard Error of Regression: " + Statistics.computeSteyx(values, values2));
        
        // T-Distribution and T-Test
        System.out.println("T-Distribution Probability: " + Statistics.computeTDist(2.1, 10));
        System.out.println("Two-Tailed T-Distribution Probability: " + Statistics.computeTDist2T(2.1, 10));
        System.out.println("Right-Tailed T-Distribution Probability: " + Statistics.computeTDistRT(2.1, 10));
        System.out.println("T-Test Statistic: " + Statistics.computeTTest(values, values2));

        // Trend Calculation
        double[] newX = {5, 10, 15};
        System.out.println("Trend Prediction: " + Arrays.toString(Statistics.computeTrend(values, values2, newX)));

        // Trimmed Mean
        System.out.println("Trimmed Mean: " + Statistics.computeTrimMean(values, 0.2));

        // Variance calculations
        System.out.println("Population Variance: " + Statistics.computeVarP(values));
        System.out.println("Sample Variance: " + Statistics.computeVarS(values));
        System.out.println("Sample Variance (Including Non-Numeric Values): " + Statistics.computeVara(mixedValues));
        System.out.println("Population Variance (Including Non-Numeric Values): " + Statistics.computeVarpa(mixedValues));

        // Weibull Distribution
        System.out.println("Weibull PDF: " + Statistics.computeWeibullDist(2, 1.5, 3, false));
        System.out.println("Weibull CDF: " + Statistics.computeWeibullDist(2, 1.5, 3, true));
        
        // Z-Test
        System.out.println("Z-Test Result: " + Statistics.computeZTest(values, 20)); 



    }
}
