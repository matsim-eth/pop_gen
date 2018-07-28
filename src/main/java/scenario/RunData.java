package scenario;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.nd4j.linalg.api.ndarray.INDArray;

import scenario.analysis.HeterogeneityLoss;
import scenario.analysis.SRMSE;
import scenario.bn.BNAlgorithm;
import scenario.bn.BNGraph;
import scenario.bn.BNGraphFinder;
import scenario.bn.BNGraphGenerator;
import scenario.bn.BNProblem;
import scenario.data.CSVReader;
import scenario.data.CountsGenerator;
import scenario.gibbs.GibbsAlgorithm;
import scenario.gibbs.GibbsSampler;
import scenario.gibbs.problem.GibbsProblemFromCounts;
import scenario.ipf.IPFAlgorithm;
import scenario.ipf.IPFSampler;
import scenario.ipf.problem.IPFProblemFromCounts;

public class RunData {
	static private Sampler createIPFSampler(INDArray counts, INDArray reducedCounts, Random random) {
		IPFProblemFromCounts problem = new IPFProblemFromCounts(counts, new int[] { 1 });
		IPFAlgorithm algorithm = new IPFAlgorithm(problem);
		algorithm.setWeights(reducedCounts.add(0.001));

		for (int i = 0; i < 100; i++)
			algorithm.runOneIteration();

		IPFSampler sampler = new IPFSampler(algorithm.getWeights(), random);

		return sampler;
	}

	static private Sampler createGibbsSampler(INDArray reducedCounts, Random random) {
		GibbsProblemFromCounts problem = new GibbsProblemFromCounts(reducedCounts);
		problem.chooseRandomInitialSample(random);

		GibbsAlgorithm algorithm = new GibbsAlgorithm(problem, random);
		GibbsSampler sampler = new GibbsSampler(algorithm, 10, 10000);

		return sampler;
	}

	static private Sampler createBNSampler(List<List<Integer>> reducedData, INDArray reducedCounts, Random random)
			throws InterruptedException {
		BNGraphGenerator graphGenerator = new BNGraphGenerator(random);

		BNGraphFinder graphFinder = new BNGraphFinder(graphGenerator, reducedCounts, reducedData, random);
		BNGraph graph = graphFinder.findGraph(1000);

		BNProblem problem = new BNProblem(reducedCounts);
		BNAlgorithm algorithm = new BNAlgorithm(graph, problem, random);
		BNAlgorithm sampler = algorithm;

		return sampler;
	}

	public static void main(String[] args) throws IOException, InterruptedException {
		File inputPath = new File(args[0]);
		List<String> columns = Arrays.asList(args[1].split(","));
		double fraction = Double.parseDouble(args[2]);
		double relativeSampleSize = Double.parseDouble(args[3]);
		String samplerName = args[4];

		Random random = new Random(0);

		List<List<Integer>> data = new CSVReader(";").load(inputPath, columns);
		Collections.shuffle(data, random);

		int reducedDataSize = (int) (fraction * data.size());
		List<List<Integer>> reducedData = data.subList(0, reducedDataSize);

		INDArray counts = new CountsGenerator().getCounts(data);
		INDArray reducedCounts = new CountsGenerator().getCounts(reducedData, data);

		Sampler sampler = null;
		int sampleSize = (int) (data.size() * relativeSampleSize);

		switch (samplerName) {
		case "ipf":
			sampler = createIPFSampler(counts, reducedCounts, random);
			break;
		case "gibbs":
			sampler = createGibbsSampler(reducedCounts, random);
			break;
		case "bn":
			sampler = createBNSampler(reducedData, reducedCounts, random);
			break;
		default:
			throw new IllegalArgumentException();
		}

		SRMSE srmse = new SRMSE(counts);
		HeterogeneityLoss heterogeneityLoss = new HeterogeneityLoss(counts);

		long lastTime = System.currentTimeMillis();

		for (int i = 0; i < sampleSize; i++) {
			int[] sample = sampler.sample();
			srmse.addSample(sample);
			heterogeneityLoss.addSample(sample);

			if (System.currentTimeMillis() - lastTime > 1000) {
				lastTime = System.currentTimeMillis();
				System.out.println(
						i + "/" + sampleSize + " # SRMSE: " + srmse.compute() + "   L: " + heterogeneityLoss.compute());
			}
		}

		System.out.println("Final # SRMSE: " + srmse.compute() + "   L: " + heterogeneityLoss.compute());
	}
}
