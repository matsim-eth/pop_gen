package scenario.bn;

import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

import org.nd4j.linalg.api.ndarray.INDArray;

public class BNGraphFinder {
	final private BNGraphGenerator generator;
	final private INDArray counts;
	final private List<List<Integer>> data;
	final private Random random;

	public BNGraphFinder(BNGraphGenerator generator, INDArray counts, List<List<Integer>> data, Random random) {
		this.generator = generator;
		this.counts = counts;
		this.data = data;
		this.random = random;
	}

	enum CRITERION {
		LOG_LIKELIHOOD, AIC, BIC
	};

	class Finder {
		public BNGraph bestGraph = null;
		public double bestScore = Double.NEGATIVE_INFINITY;

		public boolean propose(BNGraph graph, double score) {
			if (score > bestScore) {
				bestGraph = graph;
				bestScore = score;
				return true;
			}

			return false;
		}
	}

	public BNGraph findGraph(int numberOfIterations) throws InterruptedException {
		List<Thread> threads = new LinkedList<>();
		AtomicInteger iterations = new AtomicInteger(0);
		Finder finder = new Finder();

		for (int k = 0; k < Runtime.getRuntime().availableProcessors(); k++) {
			threads.add(new Thread(() -> {
				while (iterations.getAndIncrement() <= numberOfIterations) {
					BNGraph graph = null;

					synchronized (generator) {
						graph = generator.generate(data.get(0).size());
					}

					BNProblem problem = new BNProblem(counts);
					BNAlgorithm algorithm = new BNAlgorithm(graph, problem, random);

					int numberOfVariables = graph.getNumberOfVariables();
					CRITERION criterion = CRITERION.LOG_LIKELIHOOD;

					double logLikelihood = algorithm.computeLogLikelihood(data);
					double aic = 2 * numberOfVariables - 2 * logLikelihood;
					double bic = 2 * Math.log(numberOfVariables) - 2 * logLikelihood;

					double score = Double.NaN;

					switch (criterion) {
					case AIC:
						score = -aic;
					case BIC:
						score = -bic;
					case LOG_LIKELIHOOD:
						score = logLikelihood;
					}

					synchronized (finder) {
						if (finder.propose(graph, score)) {
							String info = "";
							info += "AIC" + (criterion.equals(CRITERION.AIC) ? "* " : " ") + " = " + aic;
							info += ", BIC" + (criterion.equals(CRITERION.BIC) ? "* " : " ") + " = " + bic;
							info += ", LOGLIKELIHOOD" + (criterion.equals(CRITERION.LOG_LIKELIHOOD) ? "* " : " ")
									+ " = " + logLikelihood;
							info += ", Graph: " + graph;
							System.out.println(info);
						}
					}
				}
			}));
		}

		threads.add(new Thread(() -> {
			try {
				int iteration = -1;

				do {

					Thread.sleep(1000);

					int previous = iteration;
					iteration = iterations.get();

					if (iteration > previous) {
						System.out.println(iterations.get() + " / " + numberOfIterations);
					}
				} while (iteration < numberOfIterations);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}));

		threads.forEach(t -> t.start());

		for (Thread thread : threads) {
			thread.join();
		}

		return finder.bestGraph;
	}
}
