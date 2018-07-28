package scenario.bn;

import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.Set;

public class BNGraphGenerator {
	final private Random random;
	final private Set<String> tabuList = Collections.synchronizedSet(new HashSet<>());

	public BNGraphGenerator(Random random) {
		this.random = random;
	}

	public BNGraph generate(int numberOfVariables) {
		BNGraph graph = null;

		do {
			// First generate a random ordering of variables
			List<Integer> ordering = new LinkedList<>();

			for (int i = 0; i < numberOfVariables; i++) {
				ordering.add(i);
			}

			Collections.shuffle(ordering);

			// Second build the connections. Only downstream variables can be attached.
			List<Set<Integer>> connections = new LinkedList<>();

			for (int i = 0; i < numberOfVariables - 1; i++) {
				List<Integer> possibleConnections = new LinkedList<>();
				for (int j = i + 1; j < numberOfVariables; j++) {
					possibleConnections.add(j);
				}

				Set<Integer> selectedConnections = new HashSet<>();

				// Select random number of conncetions
				int remaining = random.nextInt(possibleConnections.size() + 1);

				while (remaining > 0) {
					// Add random connection from the possible ones
					selectedConnections.add(possibleConnections.remove(random.nextInt(remaining)));
					remaining--;
				}

				connections.add(selectedConnections);
			}

			connections.add(new HashSet<>());

			graph = new BNGraph(numberOfVariables, ordering, connections);
		} while (tabuList.contains(graph.toString()));

		tabuList.add(graph.toString());

		return graph;
	}
}
