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

public class TestNetworkChoice {
	static public void main(String[] args) throws IOException, InterruptedException {
		List<List<Integer>> data = new CSVReader(";").load(new File("../data/alter_gesl.csv"),
				Arrays.asList("alter", "gesl"));
		Collections.shuffle(data);

		List<List<Integer>> reducedData = data.subList(0, 1000);

		INDArray counts = new CountsGenerator().getCounts(data);
		INDArray reducedCounts = new CountsGenerator().getCounts(reducedData, data);

		Random random = new Random(0);
		BNGraphGenerator graphGenerator = new BNGraphGenerator(random);

		BNGraphFinder graphFinder = new BNGraphFinder(graphGenerator, reducedCounts, reducedData, random);
		BNGraph graph = graphFinder.findGraph(100);
	}
}
