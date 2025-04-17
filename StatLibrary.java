package org.statlib.core;
import java.util.*;
import java.util.function.Predicate;

    public class StatLibrary {

        public static ArrayList<Double> generateRandomNumbers() {
            ArrayList<Double> randomNumbers = new ArrayList<>();
            Random random = new Random();
            for (int i = 0; i < 1000; i++) {
                randomNumbers.add(random.nextDouble() * 100);
            }
            return randomNumbers;
        }


//Returns the average of the absolute deviations of data points from their mean
        public static double computeAverageDeviation(double[] values) {
            double mean = computeAverage(values);
            double sum = 0;
            for (double v : values) {
                sum += Math.abs(v - mean);
            }
            return sum / values.length;
        }
//Returns the average of its arguments
        public static double computeAverage(double[] values) {
            double sum = 0;
            for (double v : values) {
                sum += v;
            }
            return sum / values.length;
        }

//Returns the average of its arguments and includes evaluation of text and logical values
        public static double computeArguments(Object[] values) {
            double sum = 0;
            int count = 0;
            for (Object v : values) {
                if (v instanceof Number) {
                    sum += ((Number) v).doubleValue();
                } else if (v instanceof Boolean) {
                    sum += ((Boolean) v) ? 1 : 0; // TRUE = 1, FALSE = 0
                }
                count++;
            }
            return sum / count;
        }

//Returns the average for the cells specified by a given criterion
        public static double computeAverageCells(double[] values, String operator, double threshold) {
            double sum = 0;
            int count = 0;
            for (double v : values) {
                if (evaluateCondition(v, operator, threshold)) {
                    sum += v;
                    count++;
                }
            }
            return (count == 0) ? 0 : sum / count;
        }

        private static boolean evaluateCondition(double value, String operator, double threshold) {
            return switch (operator) {
                case ">" -> value > threshold;
                case "<" -> value < threshold;
                case ">=" -> value >= threshold;
                case "<=" -> value <= threshold;
                case "==" -> value == threshold;
                case "!=" -> value != threshold;
                default -> false;
            };
        }
    }
//Returns the BetaFunction cumulative distribution function
        public static double betaFunction(double x, double y) {
            return (gammaFunction(x) * gammaFunction(y)) / gammaFunction(x + y);
        }

        public static double GammaFunction(double n) {
            if (n == 1) return 1;
            if (n == 0.5) return Math.sqrt(Math.PI);
            return (n - 1) * gammaFunction(n - 1);
        }

        public static double computeBetaDist(double x, double alpha, double beta) {
            return (Math.pow(x, alpha - 1) * Math.pow(1 - x, beta - 1)) / betaFunction(alpha, beta);
        }
//Returns the inverse of the cumulative distribution function for a specified BetaFunction distribution

        public static double computeBetaInv(double p, double alpha, double beta) {
            double x = 0.5;
            double step = 0.25;
            while (Math.abs(computeBetaDist(x, alpha, beta) - p) > 1e-6) {
                if (computeBetaDist(x, alpha, beta) < p) {
                    x += step;
                } else {
                    x -= step;
                }
                step /= 2;
            }
            return x;
        }


        public static double factorial(int n) {
            if (n == 0 || n == 1) return 1;
            return n * factorial(n - 1);
        }
    //Returns the individual term binomial distribution probability
        public static double computeBinomDist(int x, int n, double p) {
            double comb = factorial(n) / (factorial(x) * factorial(n - x));
            return comb * Math.pow(p, x) * Math.pow(1 - p, n - x);
        }

 //Returns the probability of a trial result using a binomial distribution

        public static double computeBinomDistRange(int n, double p, int x1, int x2) {
            double sum = 0;
            for (int x = x1; x <= x2; x++) {
                sum += computeBinomDist(x, n, p);
            }
            return sum;
        }
 //Returns the smallest value for which the cumulative binomial distribution is less than or equal to a criterion value

        public static int computeBinomInv(int n, double p, double target) {
            double sum = 0;
            int x = 0;
            while (sum < target && x <= n) {
                sum += computeBinomDist(x, n, p);
                x++;
            }
            return x - 1;
        }


        public static double gammaFunctionChiSqDist(double x) {
            if (x == 1) return 1;
            if (x == 0.5) return Math.sqrt(Math.PI);
            return (x - 1) * gammaFunctionChiSqDist(x - 1);
        }

        public static double chiSqCDF(double x, double df) {
            if (x < 0) return 0;
            double gamma = gammaFunctionChiSqDist(df / 2);
            return (1 / (Math.pow(2, df / 2) * gamma)) * Math.pow(x, (df / 2) - 1) * Math.exp(-x / 2);
        }

//   Returns the one-tailed probability of the chi-squared distribution

        public static double computeChiSqDistRT(double x, double df) {
            return 1 - chiSqCDF(x, df);
        }
 //Returns the cumulative BetaFunction probability density function
            public static double computeChiSqInv(double p, double df) {
            double x = df; // Initial guess
            double step = df / 2;
            while (Math.abs(chiSqCDF(x, df) - p) > 1e-6) {
                if (chiSqCDF(x, df) < p) {
                    x += step;
                } else {
                    x -= step;
                }
                step /= 2;
            }
            return x;
        }
 //Returns the inverse of the one-tailed probability of the chi-squared distribution

        public static double computeChiSqInvRT(double p, double df) {
            return computeChiSqInv(1 - p, df);
        }

 //Returns the test for independence
        public static double computeChiSqTest(double[][] observed, double[][] expected) {
            double chiSq = 0;
            for (int i = 0; i < observed.length; i++) {
                for (int j = 0; j < observed[i].length; j++) {
                    chiSq += Math.pow(observed[i][j] - expected[i][j], 2) / expected[i][j];
                }
            }
            return chiSq;
        }
//Returns the confidence interval for a population mean

    public static double computeConfidenceNorm(double alpha, double stdDev, int sampleSize) {
            double z = 1.96; // Approximate Z-score for 95% confidence
            return z * (stdDev / Math.sqrt(sampleSize));
        }

//Returns the confidence interval for a population mean, using a Student's t distribution
        public static double computeConfidenceT(double alpha, double stdDev, int sampleSize) {
            double t = 2.0; // Approximate t-score for small samples (use tables for accuracy)
            return t * (stdDev / Math.sqrt(sampleSize));
        }
//Returns the correlation coefficient between two data sets
//here
        public static double computeCorrel(double[] x, double[] y) {
            if (x.length != y.length) throw new IllegalArgumentException("Arrays must be of equal length.");

            int n = x.length;
            double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;

            for (int i = 0; i < n; i++) {
                sumX += x[i];
                sumY += y[i];
                sumXY += x[i] * y[i];
                sumX2 += x[i] * x[i];
                sumY2 += y[i] * y[i];
            }

            double numerator = (n * sumXY) - (sumX * sumY);
            double denominator = Math.sqrt(((n * sumX2) - (sumX * sumX)) * ((n * sumY2) - (sumY * sumY)));

            return (denominator == 0) ? 0 : numerator / denominator;
        }
//Counts how many numbers are in the list of arguments

        public static int computeCount(Object[] values) {
            int count = 0;
            for (Object v : values) {
                if (v instanceof Number) {
                    count++;
                }
            }
            return count;
        }
 //Counts how many values are in the list of arguments

        public static int computeCountA(Object[] values) {
            int count = 0;
            for (Object v : values) {
                if (v != null && !v.toString().trim().isEmpty()) {
                    count++;
                }
            }
            return count;
        }

//Counts the number of blank cells in the argument range
        public static int computeCountBlank(String[] values) {
            int count = 0;
            for (String v : values) {
                if (v == null || v.trim().isEmpty()) {
                    count++;
                }
            }
            return count;
        }



 //Counts the number of cells that meet the criteria you specify in the argument
        public static <T> int computeCountIf(T[] values, Predicate<T> condition) {
            int count = 0;
            for (T v : values) {
                if (condition.test(v)) {
                    count++;
                }
            }
            return count;
        }

 //Counts the number of cells that meet multiple criteria
        public static <T> int computeCountIfS(T[] values, Predicate<T>[] conditions) {
            int count = 0;
            for (T v : values) {
                boolean matches = true;
                for (Predicate<T> condition : conditions) {
                    if (!condition.test(v)) {
                        matches = false;
                        break;
                    }
                }
                if (matches) count++;
            }
            return count;
        }

//Returns covariance, the average of the products of paired deviations
        public static double computeCovarianceP(double[] x, double[] y) {
            if (x.length != y.length) throw new IllegalArgumentException("Arrays must be of equal length.");

            int n = x.length;
            double meanX = 0, meanY = 0;
            for (int i = 0; i < n; i++) {
                meanX += x[i];
                meanY += y[i];
            }
            meanX /= n;
            meanY /= n;

            double sum = 0;
            for (int i = 0; i < n; i++) {
                sum += (x[i] - meanX) * (y[i] - meanY);
            }
            return sum / n;
        }

//Returns the sample covariance, the average of the products deviations for each data point pair in two data sets
        public static double computeCovarianceS(double[] x, double[] y) {
            return computeCovarianceP(x, y) * (x.length / (x.length - 1.0));
        }

 //Returns the sum of squares of deviations
        public static double computeDevSq(double[] values) {
            double mean = 0;
            for (double v : values) {
                mean += v;
            }
            mean /= values.length;

            double sumSq = 0;
            for (double v : values) {
                sumSq += Math.pow(v - mean, 2);
            }
            return sumSq;
        }

 //Returns the exponential distribution
        public static double computeExponDist(double x, double lambda, boolean cumulative) {
            if (cumulative) {
                return 1 - Math.exp(-lambda * x);
            } else {
                return lambda * Math.exp(-lambda * x);
            }
        }

//Returns the F probability distribution
        public static double computeFDist(double x, double d1, double d2, boolean cumulative) {
            if (x < 0) return 0;
            double num = (Math.pow(d1 * x, d1) * Math.pow(d2, d2)) / (Math.pow(d1 * x + d2, d1 + d2));
            double den = x * BetaFunction(d1 / 2, d2 / 2);
            return cumulative ? num / den : num;
        }


        public static double BetaFunction(double a, double b) {
            return (gammaFunction(a) * gammaFunction(b)) / gammaFunction(a + b);
        }

        public static double gammaFunction(double x) {
            if (x == 1) return 1;
            if (x == 0.5) return Math.sqrt(Math.PI);
            return (x - 1) * gammaFunction(x - 1);
        }

 //Returns the inverse of the F probability distribution

        public static double computeFInv(double p, double d1, double d2) {
            double x = 1.0; // Initial guess
            double step = 0.1;
            while (Math.abs(computeFDist(x, d1, d2, true) - p) > 1e-6) {
                if (computeFDist(x, d1, d2, true) < p) {
                    x += step;
                } else {
                    x -= step;
                }
                step /= 2;
            }
            return x;
        }

//Returns the result of an F-test

        public static double computeFTest(double[] sample1, double[] sample2) {
            double var1 = variance(sample1);
            double var2 = variance(sample2);
            return var1 / var2;
        }

        private static double variance(double[] data) {
            double mean = 0, sumSq = 0;
            for (double d : data) mean += d;
            mean /= data.length;
            for (double d : data) sumSq += Math.pow(d - mean, 2);
            return sumSq / (data.length - 1);
        }

//Returns the Fisher transformation
        public static double computeFisher(double x) {

    return 0.5 * Math.log((1 + x) / (1 - x));
        }

//Returns the inverse of the Fisher transformation

        public static double computeFisherInv(double y) {

    return (Math.exp(2 * y) - 1) / (Math.exp(2 * y) + 1);
        }

//Returns a value along a linear trend
        public static double computeForecast(double x, double[] knownX, double[] knownY) {
            int n = knownX.length;
            double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;

            for (int i = 0; i < n; i++) {
                sumX += knownX[i];
                sumY += knownY[i];
                sumXY += knownX[i] * knownY[i];
                sumX2 += knownX[i] * knownX[i];
            }

            double slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
            double intercept = (sumY - slope * sumX) / n;
            return slope * x + intercept;
        }

//Calculates a future value based on existing values using the Exponential Triple Smoothing (ETS) algorithm
        public static double computeForecastETS(double[] values, int period) {
            int n = values.length;
            double alpha = 0.5; // Smoothing factor (adjustable)

            double[] smoothed = new double[n];
            smoothed[0] = values[0];

            for (int i = 1; i < n; i++) {
                smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1];
            }

            return smoothed[n - 1]; // Forecast next value
        }

//Returns a confidence interval for the forecast value at the specified target date
        public static double computeForecastETSConfInt(double[] values, int period, double confidenceLevel) {
            double forecast = computeForecastETS(values, period);
            double stdDev = standardDeviation(values);

            double marginOfError = confidenceLevel * stdDev / Math.sqrt(values.length);
            return marginOfError;
        }

        private static double standardDeviation(double[] values) {
            double mean = 0, sumSq = 0;
            for (double v : values) mean += v;
            mean /= values.length;
            for (double v : values) sumSq += Math.pow(v - mean, 2);
            return Math.sqrt(sumSq / values.length);
        }
//Returns the length of the repetitive pattern detected for the specified time series

        public static int computeForecastETSSeasonality(double[] values) {
            int n = values.length;
            int bestPeriod = 1;
            double minError = Double.MAX_VALUE;

            for (int p = 2; p <= n / 2; p++) {
                double error = 0;
                for (int i = p; i < n; i++) {
                    error += Math.abs(values[i] - values[i - p]);
                }
                error /= (n - p);
                if (error < minError) {
                    minError = error;
                    bestPeriod = p;
                }
            }
            return bestPeriod;
        }
//Returns a statistical value as a result of time series forecasting

        public static double computeForecastETSStat(double[] values) {
            int n = values.length;
            double[] smoothed = new double[n];

            smoothed[0] = values[0];
            for (int i = 1; i < n; i++) {
                smoothed[i] = 0.5 * values[i] + 0.5 * smoothed[i - 1];
            }

            double mse = 0;
            for (int i = 0; i < n; i++) {
                mse += Math.pow(values[i] - smoothed[i], 2);
            }
            return mse / n;
        }
//Calculates a future value by using existing values, using linear regression.

        public static double computeForecastLinear(double x, double[] knownX, double[] knownY) {
            return computeForecast(x, knownX, knownY);
        }

//Returns a frequency distribution as a vertical array
        public static int[] computeFrequency(double[] data, double[] bins) {
            int[] frequency = new int[bins.length + 1];
            Arrays.fill(frequency, 0);

            for (double value : data) {
                int i = 0;
                while (i < bins.length && value > bins[i]) {
                    i++;
                }
                frequency[i]++;
            }
            return frequency;
        }

//Returns the Gamma function value
        public static double computeGamma(double x) {
            if (x == 1) return 1;
            if (x == 0.5) return Math.sqrt(Math.PI);
            return (x - 1) * computeGamma(x - 1);
        }

//Returns the gamma distribution
        public static double computeGammaDist(double x, double alpha, double beta, boolean cumulative) {
            if (cumulative) {
                return incompleteGamma(x / beta, alpha);
            } else {
                return (Math.pow(x, alpha - 1) * Math.exp(-x / beta)) /
                        (Math.pow(beta, alpha) * computeGamma(alpha));
            }
        }

        private static double incompleteGamma(double x, double alpha) {
            double sum = 1.0 / alpha;
            double term = 1.0 / alpha;
            for (int n = 1; n < 100; n++) {
                term *= x / (alpha + n);
                sum += term;
            }
            return sum * Math.exp(-x) * Math.pow(x, alpha);
        }

//Returns the inverse of the gamma cumulative distribution
        public static double computeGammaInv(double p, double alpha, double beta) {
            double x = alpha * beta; // Initial estimate
            double tolerance = 1e-6;

            while (true) {
                double fx = computeGammaDist(x, alpha, beta, true) - p;
                double dfx = (computeGammaDist(x + tolerance, alpha, beta, true) -
                        computeGammaDist(x, alpha, beta, true)) / tolerance;

                double newX = x - fx / dfx;
                if (Math.abs(newX - x) < tolerance) return newX;
                x = newX;
            }
        }

//Returns the natural logarithm of the gamma function, G(x)
       public static double computeGammaLn(double x) {
            return Math.log(computeGamma(x));
        }
//Returns the natural logarithm of the gamma function, G(x)

        public static double computeGammaLnPrecise(double x) {
            return computeGammaLn(x);
        }

//Returns 0.5 less than the standard normal cumulative distribution
        public static double computeGauss(double z) {
            return computeNormalDist(z, 0, 1, true) - 0.5;
        }


            public static double computeNormalDist(double x, double mean, double stdDev, boolean cumulative) {
                double z = (x - mean) / (stdDev * Math.sqrt(2));
                return cumulative ? 0.5 * (1 + ERF(z)) : Math.exp(-z * z) / (stdDev * Math.sqrt(2 * Math.PI));
            }

            private static double ERF(double x) {
                double t = 1 / (1 + 0.3275911 * Math.abs(x));
                double poly = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
                double approx = 1 - poly * Math.exp(-x * x);
                return x >= 0 ? approx : -approx;
            }

//Returns the geometric mean

        public static double computeGeomean(double[] values) {
            double product = 1.0;
            for (double v : values) {
                product *= v;
            }
            return Math.pow(product, 1.0 / values.length);
        }

//Returns values along an exponential trend
        public static double[] computeGrowth(double[] knownY, double[] knownX, double[] newX) {
            int n = knownX.length;
            double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;

            for (int i = 0; i < n; i++) {
                sumX += knownX[i];
                sumY += Math.log(knownY[i]);
                sumXY += knownX[i] * Math.log(knownY[i]);
                sumX2 += knownX[i] * knownX[i];
            }

            double b = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
            double a = Math.exp((sumY - b * sumX) / n);

            double[] results = new double[newX.length];
            for (int i = 0; i < newX.length; i++) {
                results[i] = a * Math.exp(b * newX[i]);
            }
            return results;
        }

//Returns the harmonic mean
        public static double computeHarmean(double[] values) {
            double sumReciprocal = 0;
            for (double v : values) {
                sumReciprocal += 1.0 / v;
            }
            return values.length / sumReciprocal;
        }

//Returns the hypergeometric distribution
        public static double computeHypergeomDist(int x, int M, int n, int N, boolean cumulative) {
            // x = number of successes in sample
            // M = population size
            // n = number of successes in population
            // N = sample size
            // cumulative = whether to computeBinomDist cumulative probability

            double result = 0;
            for (int k = x; k <= N; k++) {
                result += binomialCoeff(n, k) * binomialCoeff(M - n, N - k) / binomialCoeff(M, N);
            }

            return result;
        }

        private static int binomialCoeff(int n, int k) {
            if (k > n) return 0;
            if (k == 0 || k == n) return 1;
            return binomialCoeff(n - 1, k - 1) + binomialCoeff(n - 1, k);
        }

//Returns the intercept of the linear regression line
        public static double computeIntercept(double[] x, double[] y) {
            double meanX = Mean(x);
            double meanY = Mean(y);
            double slope = Slope(x, y, meanX, meanY);
            return meanY - slope * meanX;
        }

        private static double Mean(double[] values) {
            double sum = 0;
            for (double value : values) sum += value;
            return sum / values.length;
        }

        private static double Slope(double[] x, double[] y, double meanX, double meanY) {
            double numerator = 0;
            double denominator = 0;
            for (int i = 0; i < x.length; i++) {
                numerator += (x[i] - meanX) * (y[i] - meanY);
                denominator += (x[i] - meanX) * (x[i] - meanX);
            }
            return numerator / denominator;
        }

    //Returns the kurtosis of a data set

        public static double computeKurtosis(double[] data) {
            double mean = MEan(data);
            double n = data.length;
            double sumFourthMoment = 0;
            double sumSquaredMoment = 0;

            for (double x : data) {
                sumFourthMoment += Math.pow(x - mean, 4);
                sumSquaredMoment += Math.pow(x - mean, 2);
            }

            double kurtosis = (n * (n + 1) * sumFourthMoment) / ((n - 1) * (n - 2) * (n - 3) * Math.pow(sumSquaredMoment / n, 2));
            kurtosis -= (3 * Math.pow(n - 1, 2)) / ((n - 2) * (n - 3));
            return kurtosis;
        }

        private static double MEan(double[] values) {
            double sum = 0;
            for (double value : values) sum += value;
            return sum / values.length;
        }

 //Returns the kth largest value in a data set
        public static double computeLarge(double[] data, int k) {
            Arrays.sort(data);
            return data[data.length - k];
        }

 //Returns the parameters of a linear trend
        public static double[] computeLinest(double[] x, double[] y) {
            double meanX = mean(x);
            double meanY = mean(y);
            double slope = slope(x, y, meanX, meanY);
            double intercept = meanY - slope * meanX;
            return new double[]{slope, intercept};
        }

        private static double mean(double[] values) {
            double sum = 0;
            for (double value : values) sum += value;
            return sum / values.length;
        }

        private static double slope(double[] x, double[] y, double meanX, double meanY) {
            double numerator = 0;
            double denominator = 0;
            for (int i = 0; i < x.length; i++) {
                numerator += (x[i] - meanX) * (y[i] - meanY);
                denominator += (x[i] - meanX) * (x[i] - meanX);
            }
            return numerator / denominator;
        }

//Returns the parameters of an exponential trend
        public static double[] computeLogest(double[] x, double[] y) {
            double sumLogY = 0;
            double sumX = 0;
            double sumXLogY = 0;
            double sumX2 = 0;

            for (int i = 0; i < x.length; i++) {
                sumLogY += Math.log(y[i]);
                sumX += x[i];
                sumXLogY += x[i] * Math.log(y[i]);
                sumX2 += x[i] * x[i];
            }

            double slope = (x.length * sumXLogY - sumX * sumLogY) / (x.length * sumX2 - sumX * sumX);
            double intercept = (sumLogY - slope * sumX) / x.length;

            return new double[]{Math.exp(intercept), slope};
        }

//Returns the cumulative lognormal distribution
        public static double computeLogNormDist(double x, double mean, double stddev, boolean cumulative) {
            double logX = Math.log(x);
            double exponent = -Math.pow(logX - mean, 2) / (2 * stddev * stddev);
            double result = (1 / (x * stddev * Math.sqrt(2 * Math.PI))) * Math.exp(exponent);

            if (cumulative) {
                return 0.5 * (1 + Erf((logX - mean) / (stddev * Math.sqrt(2))));
            }
            return result;
        }

        private static double Erf(double z) {
            double t = 1.0 / (1.0 + 0.5 * Math.abs(z));
            double tau = t * Math.exp(-z * z - 1.26551223
                    + 1.00002368 * t + 0.37409196 * t * t
                    + 0.09678418 * t * t * t - 0.18628806 * t * t * t * t
                    + 0.27886807 * t * t * t * t * t - 1.13520398 * t * t * t * t * t * t);
            return z >= 0.0 ? 1 - tau : tau - 1;
        }

//Returns the inverse of the lognormal cumulative distribution
        public static double computeLogNormInv(double p, double mean, double stddev) {
            double z = InverseErf(2 * p - 1);
            return Math.exp(mean + stddev * z);
        }

        private static double InverseErf(double p) {
            double a = 0.147;
            double ln = Math.log(1 - p * p);
            double result = Math.sqrt(Math.sqrt(Math.pow(2 / (Math.PI * a) + ln / 2, 2) - ln / a) - (2 / (Math.PI * a) + ln / 2));
            return (p < 0) ? -result : result;
        }

//Returns the maximum value in a list of arguments, ignoring logical values and text
        public static double computeMax(double[] values) {
            double max = values[0];
            for (double value : values) {
                if (value > max) max = value;
            }
            return max;
        }

 //Returns the maximum value in a list of arguments, including logical values and text   public class MaxA {
        public static double computeMaxA(Object[] values) {
            double max = Double.NEGATIVE_INFINITY;
            for (Object value : values) {
                if (value instanceof Number) {
                    max = Math.max(max, ((Number) value).doubleValue());
                } else if (value instanceof Boolean) {
                    max = Math.max(max, (Boolean) value ? 1 : 0);
                }
            }
            return max;
        }
 //Returns the maximum value among cells specified by a given set of conditions or criteria.

        public static double computeMaxIfs(double[] values, boolean[] criteria) {
            double max = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < values.length; i++) {
                if (criteria[i]) {
                    max = Math.max(max, values[i]);
                }
            }
            return max;
        }

//Returns the median of the given numbers
        public static double computeMedian(double[] values) {
            Arrays.sort(values);
            int n = values.length;
            if (n % 2 == 1) {
                return values[n / 2];
            } else {
                return (values[n / 2 - 1] + values[n / 2]) / 2.0;
            }
        }

//Returns the minimum value in a list of arguments, ignoring logical values and text
        public static double computeMin(double[] values) {
            double min = values[0];
            for (double value : values) {
                if (value < min) min = value;
            }
            return min;
        }

//Returns the minimum value in a list of arguments, including logical values and text
        public static double computeMinA(Object[] values) {
            double min = Double.POSITIVE_INFINITY;
            for (Object value : values) {
                if (value instanceof Number) {
                    min = Math.min(min, ((Number) value).doubleValue());
                } else if (value instanceof Boolean) {
                    min = Math.min(min, (Boolean) value ? 1 : 0);
                }
            }
            return min;
        }

 //Returns the minimum value among cells specified by a given set of conditions or criteria.
        public static double computeMinIfs(double[] values, boolean[] criteria) {
            double min = Double.POSITIVE_INFINITY;
            for (int i = 0; i < values.length; i++) {
                if (criteria[i]) {
                    min = Math.min(min, values[i]);
                }
            }
            return min;
        }

//Returns a vertical array of the most frequently occurring, or repetitive values in an array or range of data
        public static double computeModeSngl(double[] values) {
            Map<Double, Integer> frequencyMap = new HashMap<>();
            for (double value : values) {
                frequencyMap.put(value, frequencyMap.getOrDefault(value, 0) + 1);
            }

            return Collections.max(frequencyMap.entrySet(), Map.Entry.comparingByValue()).getKey();
        }

//Returns the negative binomial distribution
        public static double computeNegBinomDist(int k, int r, double p, boolean cumulative) {
            // k = number of failures before the rth success
            // r = number of successes
            // p = probability of success in each trial

            double result = (factorial(k + r - 1) / (factorial(k) * factorial(r - 1))) * Math.pow(p, r) * Math.pow(1 - p, k);

            if (cumulative) {
                return cumulativeNegativeBinomial(k, r, p);
            }

            return result;
        }

        private static double cumulativeNegativeBinomial(int k, int r, double p) {
            double sum = 0;
            for (int i = 0; i <= k; i++) {
                sum += computeNegBinomDist(i, r, p, false);
            }
            return sum;



        }
//Returns the normal cumulative distribution

        public static double computeNormDist(double z, double mean, double stddev, boolean cumulative) {
            double exponent = -Math.pow(z - mean, 2) / (2 * stddev * stddev);
            double result = (1 / (stddev * Math.sqrt(2 * Math.PI))) * Math.exp(exponent);

            if (cumulative) {
                return 0.5 * (1 + erf((z - mean) / (stddev * Math.sqrt(2))));
            }

            return result;
        }

        private static double erf(double z) {
            double t = 1.0 / (1.0 + 0.5 * Math.abs(z));
            double tau = t * Math.exp(-z * z - 1.26551223
                    + 1.00002368 * t + 0.37409196 * t * t
                    + 0.09678418 * t * t * t - 0.18628806 * t * t * t * t
                    + 0.27886807 * t * t * t * t * t - 1.13520398 * t * t * t * t * t * t);
            return z >= 0.0 ? 1 - tau : tau - 1;
        }

//Returns the inverse of the normal cumulative distribution

        public static double computeNormInv(double p, double mean, double stddev) {
            // p = cumulative probability
            double z = inverseerf(2 * p - 1);
            return mean + stddev * z;
        }

        private static double inverseerf(double p) {
            double a = 0.147;
            double ln = Math.log(1 - p * p);
            double result = Math.sqrt(Math.sqrt(Math.pow(2 / (Math.PI * a) + ln / 2, 2) - ln / a) - (2 / (Math.PI * a) + ln / 2));
            return (p < 0) ? -result : result;
        }

//Returns the standard normal cumulative distribution
        public static double computeormSDist(double z, boolean cumulative) {
            if (cumulative) {
                return 0.5 * (1 + erf(z / Math.sqrt(2)));
            } else {
                return (1 / Math.sqrt(2 * Math.PI)) * Math.exp(-0.5 * z * z);
            }
        }



 //Returns the inverse of the standard normal cumulative distribution
        public static double computeNormSInv(double p) {
            return Math.sqrt(2) * inverseErf(2 * p - 1);
        }

        private static double inverseErf(double p) {
            double a = 0.147;
            double ln = Math.log(1 - p * p);
            double result = Math.sqrt(Math.sqrt(Math.pow(2 / (Math.PI * a) + ln / 2, 2) - ln / a) - (2 / (Math.PI * a) + ln / 2));
            return (p < 0) ? -result : result;
        }

 //Returns the Pearson product moment correlation coefficient
        public static double computePearson(double[] x, double[] y) {
            int n = x.length;
            double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;
            for (int i = 0; i < n; i++) {
                sumX += x[i];
                sumY += y[i];
                sumXY += x[i] * y[i];
                sumX2 += x[i] * x[i];
                sumY2 += y[i] * y[i];
            }
            double numerator = n * sumXY - sumX * sumY;
            double denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
            return numerator / denominator;
        }

//Returns the k-th percentile of values in a range, where k is in the range 0..1, exclusive
        public static double computePercentileExc(double[] values, double k) {
            if (k <= 0 || k >= 1) throw new IllegalArgumentException("k must be between 0 and 1, exclusive.");
            Arrays.sort(values);
            double rank = k * (values.length - 1);
            int lowerIndex = (int) Math.floor(rank);
            int upperIndex = (int) Math.ceil(rank);

            if (lowerIndex == upperIndex) {
                return values[lowerIndex];
            }

            return values[lowerIndex] + (values[upperIndex] - values[lowerIndex]) * (rank - lowerIndex);
        }
//Returns the k-th percentile of values in a range

        public static double computePercentileInc(double[] values, double k) {
            if (k < 0 || k > 1) throw new IllegalArgumentException("k must be between 0 and 1 inclusive.");
            Arrays.sort(values);
            double rank = k * (values.length - 1);
            int lowerIndex = (int) Math.floor(rank);
            int upperIndex = (int) Math.ceil(rank);

            if (lowerIndex == upperIndex) {
                return values[lowerIndex];
            }

            return values[lowerIndex] + (values[upperIndex] - values[lowerIndex]) * (rank - lowerIndex);
        }
//Returns the rank of a value in a data set as a percentage (0..1, exclusive) of the data set
        public static double computePercentRankInc(double[] values, double x) {
            Arrays.sort(values);
            int rank = 0;
            for (double value : values) {
                if (value <= x) {
                    rank++;
                }
            }
            return rank / (double)(values.length - 1);
        }

//Returns the number of permutations for a given number of objects
        public static long computePermut(int n, int r) {
            return Factorial(n) / Factorial(n - r);
        }

        private static long Factorial(int n) {
            long result = 1;
            for (int i = 1; i <= n; i++) {
                result *= i;
            }
            return result;
        }
//Returns the number of permutations for a given number of objects (with repetitions) that can be selected from the total objects

        public static long computePermutationA(int n, int r) {
            return (long) Math.pow(n, r);
        }

 //Returns the value of the density function for a standard normal distribution

        public static double computePhi (double z) {
            return (1 / Math.sqrt(2 * Math.PI)) * Math.exp(-0.5 * z * z);
        }
 //Returns the Poisson distribution

        public static double computePoissonDist(int x, double mean, boolean cumulative) {
            double result = Math.pow(mean, x) * Math.exp(-mean) / factorial(x);
            if (cumulative) {
                double cumulativeSum = 0;
                for (int i = 0; i <= x; i++) {
                    cumulativeSum += Math.pow(mean, i) * Math.exp(-mean) / factorial(i);
                }
                return cumulativeSum;
            }
            return result;
        }



//Returns the probability that values in a range are between two limits
        public static double computeProb(double[] values, double lowerLimit, double upperLimit) {
            int count = 0;
            for (double value : values) {
                if (value >= lowerLimit && value <= upperLimit) {
                    count++;
                }
            }
            return (double) count / values.length;
        }
//Returns the quartile of the data set, based on percentile values from 0..1, exclusive

        public static double computeQuartileExc(double[] values, int quartile) {
            if (quartile < 1 || quartile > 3) throw new IllegalArgumentException("Quartile must be 1, 2, or 3.");
            Arrays.sort(values);
            double rank = quartile * (values.length - 1) / 4.0;
            int lowerIndex = (int) Math.floor(rank);
            int upperIndex = (int) Math.ceil(rank);

            if (lowerIndex == upperIndex) {
                return values[lowerIndex];
            }

            return values[lowerIndex] + (values[upperIndex] - values[lowerIndex]) * (rank - lowerIndex);
        }

//Returns the quartile of a data set
        public static double computeQuartileInc(double[] values, int quartile) {
            if (quartile < 1 || quartile > 3) throw new IllegalArgumentException("Quartile must be 1, 2, or 3.");
            Arrays.sort(values);
            double rank = quartile * (values.length - 1) / 4.0;
            int lowerIndex = (int) Math.floor(rank);
            int upperIndex = (int) Math.ceil(rank);

            if (lowerIndex == upperIndex) {
                return values[lowerIndex];
            }

            return values[lowerIndex] + (values[upperIndex] - values[lowerIndex]) * (rank - lowerIndex);
        }

//Returns the rank of a number in a list of numbers
        public static double computeRankAvg(double[] values, double x) {
            Arrays.sort(values);
            double rankSum = 0;
            int count = 0;
            for (int i = 0; i < values.length; i++) {
                if (values[i] == x) {
                    rankSum += i + 1;
                    count++;
                }
            }
            return rankSum / count;
        }

 //Returns the rank of a number in a list of numbers
        public static int computeRankEq(double[] values, double x) {
            Arrays.sort(values);
            for (int i = 0; i < values.length; i++) {
                if (values[i] == x) {
                    return i + 1;
                }
            }
            return -1;
        }


//Returns the skewness of a distribution
        public static double computeSkew(double[] values) {
            double n = values.length;
            double mean = Arrays.stream(values).average().orElse(Double.NaN);
            double variance = Arrays.stream(values).map(v -> Math.pow(v - mean, 2)).average().orElse(Double.NaN);
            double skewness = Arrays.stream(values).map(v -> Math.pow(v - mean, 3)).average().orElse(Double.NaN);
            return (n * skewness) / ((n - 1) * (n - 2) * Math.pow(variance, 1.5));
        }

        //Returns the skewness of a distribution based on a population: a characterization of the degree of asymmetry of a distribution around its mean
        public static double computeSkewP(double[] values) {
            double n = values.length;
            double mean = Arrays.stream(values).average().orElse(Double.NaN);
            double variance = Arrays.stream(values).map(v -> Math.pow(v - mean, 2)).average().orElse(Double.NaN);
            double skewness = Arrays.stream(values).map(v -> Math.pow(v - mean, 3)).average().orElse(Double.NaN);
            return (n * skewness) / ((n - 1) * (n - 2) * Math.pow(variance, 1.5));
        }

//Returns the slope of the linear regression line
        public static double computeSlope(double[] x, double[] y) {
            int n = x.length;
            double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;

            for (int i = 0; i < n; i++) {
                sumX += x[i];
                sumY += y[i];
                sumXY += x[i] * y[i];
                sumX2 += x[i] * x[i];
            }

            return (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        }

//Returns the kth smallest value in a data set
        public static double computeSmall(double[] values, int k) {
            Arrays.sort(values);
            if (k > 0 && k <= values.length) {
                return values[k - 1];
            }
            throw new IllegalArgumentException("k must be between 1 and the length of the dataset.");
        }

//Returns a normalized value
        public static double computeStandardize(double x, double mean, double stdDev) {
            if (stdDev == 0) {
                throw new IllegalArgumentException("Standard deviation cannot be zero.");
            }
            return (x - mean) / stdDev;
        }

//Calculates standard deviation based on the entire population
        public static double computeStdevP(double[] values) {
            double mean = Arrays.stream(values).average().orElse(Double.NaN);
            double variance = Arrays.stream(values)
                    .map(v -> Math.pow(v - mean, 2))
                    .average()
                    .orElse(Double.NaN);
            return Math.sqrt(variance);
        }

//Estimates standard deviation based on a sample
        public static double computeStdevS(double[] values) {
            int n = values.length;
            double mean = Arrays.stream(values).average().orElse(Double.NaN);
            double variance = Arrays.stream(values)
                    .map(v -> Math.pow(v - mean, 2))
                    .sum() / (n - 1);  // n-1 for sample standard deviation
            return Math.sqrt(variance);
        }

 //Estimates standard deviation based on a sample, including text and logical values
        public static double computeStdevA(Object[] values) {
            int n = values.length;
            double mean = Arrays.stream(values)
                    .filter(v -> v instanceof Number)
                    .mapToDouble(v -> ((Number) v).doubleValue())
                    .average()
                    .orElse(Double.NaN);
            double variance = Arrays.stream(values)
                    .filter(v -> v instanceof Number)
                    .mapToDouble(v -> Math.pow(((Number) v).doubleValue() - mean, 2))
                    .sum() / (n - 1);
            return Math.sqrt(variance);
        }

//Calculates standard deviation based on the entire population, including text and logical values
        public static double computeStdevPA (Object[] values) {
            int n = values.length;
            double mean = Arrays.stream(values)
                    .filter(v -> v instanceof Number)
                    .mapToDouble(v -> ((Number) v).doubleValue())
                    .average()
                    .orElse(Double.NaN);
            double variance = Arrays.stream(values)
                    .filter(v -> v instanceof Number)
                    .mapToDouble(v -> Math.pow(((Number) v).doubleValue() - mean, 2))
                    .sum() / n;  // Entire population
            return Math.sqrt(variance);
        }

//Returns the standard error of the predicted y-value for each x in the regression
        public static double computeSteyx(double[] x, double[] y) {
            double meanX = Arrays.stream(x).average().orElse(Double.NaN);
            double meanY = Arrays.stream(y).average().orElse(Double.NaN);
            double ssXX = 0, ssYY = 0, ssXY = 0;

            for (int i = 0; i < x.length; i++) {
                ssXX += Math.pow(x[i] - meanX, 2);
                ssYY += Math.pow(y[i] - meanY, 2);
                ssXY += (x[i] - meanX) * (y[i] - meanY);
            }

            double slope = ssXY / ssXX;
            double intercept = meanY - slope * meanX;
            double totalError = 0;
            for (int i = 0; i < x.length; i++) {
                double predictedY = slope * x[i] + intercept;
                totalError += Math.pow(y[i] - predictedY, 2);
            }

            return Math.sqrt(totalError / (x.length - 2));  // Degrees of freedom
        }

//Returns the Percentage Points (probability) for the Student t-distribution
        public static double computeTDist(double t, int degreesOfFreedom) {
            // Using a numerical approximation (for simplicity)
            double x = t / Math.sqrt(degreesOfFreedom);
            return incompleteBeta(x, degreesOfFreedom / 2.0, 0.5);
        }

        // Incomplete Beta function (approximated)
        private static double incompleteBeta(double x, double a, double b) {
            // Basic approximation (simplified form of the incomplete BetaFunction function)
            double result = 0.0;
            if (x < 0 || x > 1 || a <= 0 || b <= 0) return result;

            // Using a series expansion for the Incomplete Beta function
            result = 1 - Math.pow(x, a) / (a + b);
            return result;
        }


        public static double computeTDist2T(double t, int degreesOfFreedom) {
            // Using the symmetry of the t-distribution (two-tailed)
            double oneTailedCDF = computeTDist(t, degreesOfFreedom);
            return 2 * (1 - oneTailedCDF);
        }


        public static double computeTDistRT(double t, int degreesOfFreedom) {
            // Using 1 minus the CDF to calculate the right-tail probability
            return 1 - computeTDist(t, degreesOfFreedom);
        }



        public static double computeTTest(double[] sample1, double[] sample2) {
            double mean1 = CalculateMean(sample1);
            double mean2 = CalculateMean(sample2);
            double var1 = calculateVariance(sample1, mean1);
            double var2 = calculateVariance(sample2, mean2);

            int n1 = sample1.length;
            int n2 = sample2.length;

            double pooledStandardError = Math.sqrt((var1 / n1) + (var2 / n2));
            double tStatistic = (mean1 - mean2) / pooledStandardError;

            // For the actual p-value, you'd need to calculate the cumulative distribution.
            return tStatistic; // Placeholder for the actual p-value calculation
        }

        private static double CalculateMean(double[] sample) {
            double sum = 0;
            for (double value : sample) {
                sum += value;
            }
            return sum / sample.length;
        }

        private static double calculateVariance(double[] sample, double mean) {
            double sum = 0;
            for (double value : sample) {
                sum += Math.pow(value - mean, 2);
            }
            return sum / (sample.length - 1); // Sample variance
        }


        public static double computeTrend(double[] x, double[] y, double[] newX) {
            double meanX = CalculateMean(x);
            double meanY = CalculateMean(y);
            double ssXX = 0, ssXY = 0;

            for (int i = 0; i < x.length; i++) {
                ssXX += Math.pow(x[i] - meanX, 2);
                ssXY += (x[i] - meanX) * (y[i] - meanY);
            }

            double slope = ssXY / ssXX;
            double intercept = meanY - slope * meanX;
            double[] predictedY = new double[newX.length];

            for (int i = 0; i < newX.length; i++) {
                predictedY[i] = slope * newX[i] + intercept;
            }

            return predictedY;


        private static double calculateMean (double[] values) {
            double sum = 0;
            for (double value : values) {
                sum += value;
            }
            return sum / values.length;
        }
    }

        public static double computeTrimMean(double[] values, double[] percent) {
            if (percent < 0 || percent >= 1) {
                throw new IllegalArgumentException("Percent must be between 0 and 1.");
            }

            Arrays.sort(values);
            int trimCount = (int) (values.length * percent / 2);
            double[] trimmed = Arrays.copyOfRange(values, trimCount, values.length - trimCount);
            return CalculateMean(trimmed);
        }




        public static double computeVarP(double[] values) {
            double mean = calculateMean(values);
            double variance = 0;

            for (double value : values) {
                variance += Math.pow(value - mean, 2);
            }

            return variance / values.length; // Population variance
        }

        private static double calculateMean(double[] values) {
            double sum = 0;
            for (double value : values) {
                sum += value;
            }
            return sum / values.length;
        }


        public static double computeVarS(double[] values) {
            int n = values.length;
            double mean = CalculateMean(values);
            double variance = 0;

            for (double value : values) {
                variance += Math.pow(value - mean, 2);
            }

            return variance / (n - 1); // Sample variance
        }


        public static double computeVara(Object[] values) {
            int n = values.length;
            double sum = 0;
            int count = 0;

            for (Object value : values) {
                if (value instanceof Number) {
                    sum += ((Number) value).doubleValue();
                    count++;
                }
            }

            double mean = sum / count;
            double variance = 0;

            for (Object value : values) {
                if (value instanceof Number) {
                    variance += Math.pow(((Number) value).doubleValue() - mean, 2);
                }
            }

            return variance / (n - 1); // Sample variance
        }


        public static double computeVarpa(Object[] values) {
            int n = values.length;
            double sum = 0;
            int count = 0;

            for (Object value : values) {
                if (value instanceof Number) {
                    sum += ((Number) value).doubleValue();
                    count++;
                }
            }

            double mean = sum / count;
            double variance = 0;

            for (Object value : values) {
                if (value instanceof Number) {
                    variance += Math.pow(((Number) value).doubleValue() - mean, 2);
                }
            }

            return variance / n; // Population variance
        }


        public static double computeWeibullDist(double x, double shape, double scale, boolean cumulative) {
            if (x < 0 || shape <= 0 || scale <= 0) {
                throw new IllegalArgumentException("Invalid input parameters.");
            }

            double density = (shape / scale) * Math.pow(x / scale, shape - 1) * Math.exp(-Math.pow(x / scale, shape));

            if (cumulative) {
                // CDF (cumulative distribution function) calculation
                return 1 - Math.exp(-Math.pow(x / scale, shape));
            } else {
                return density; // PDF (probability density function)
            }
        }


        public static double computeZTest(double[] sample, double populationMean) {
            double mean = calculateMean(sample);
            double variance = calculateVariance(sample, mean);
            double standardDeviation = Math.sqrt(variance / sample.length);

            double zStatistic = (mean - populationMean) / standardDeviation;
            return calculateZProbability(zStatistic);
        }



        private static double calculateZProbability(double z) {
            // Z-table lookup or approximation needed here
            return 0.0; // Placeholder
        }



