package scenario;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.nd4j.linalg.api.ndarray.INDArray;

import scenario.bn.BNGraph;
import scenario.bn.BNGraphFinder;
import scenario.bn.BNGraphGenerator;
import scenario.data.CSVReader;
import scenario.data.CountsGenerator;

public class RunHouseholds {
	static public void main(String[] args) throws IOException, InterruptedException {
		File inputPath = new File(args[0]);

		List<String> columns = Arrays.asList("household_income", "household_cars", "household_bikes", "household_size",
				"household_max_age", "household_min_age");

		Random random = new Random(0);

		List<List<Integer>> data = new CSVReader(",").load(inputPath, columns);
		Collections.shuffle(data, random);

		INDArray counts = new CountsGenerator().getCounts(data);
		
		System.out.println("data loaded");

		BNGraphGenerator graphGenerator = new BNGraphGenerator(random);

		BNGraphFinder graphFinder = new BNGraphFinder(graphGenerator, counts, data, random);
		BNGraph graph = graphFinder.findGraph(1000);

		System.out.println("Final: " + graph);

		// BNProblem problem = new BNProblem(counts);
		// BNAlgorithm algorithm = new BNAlgorithm(graph, problem, random);
	}
}
