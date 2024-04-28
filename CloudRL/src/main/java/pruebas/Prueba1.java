package pruebas;

import com.pureedgesim.MainApplication;
import com.pureedgesim.simulationcore.Simulation;
import com.pureedgesim.tasksorchestration.DefaultOrchestrator;

public class Prueba1 extends MainApplication {
	//private static String settingsPath = "PureEdgeSim/pruebas/settings/";
	private static String settingsPath = "PureEdgeSim/pruebas/settings/";
	private static String outputPath = "PureEdgeSim/pruebas/output/";

	public static void main(String[] args) {
		Simulation sim = new Simulation();
		
		sim.setCustomOutputFolder(outputPath);
		sim.setCustomSettingsFolder(settingsPath);


		sim.setCustomEdgeOrchestrator(DefaultOrchestrator.class);
		//sim.setCustomEnergyModel(CustomEnergyModel.class);
		//sim.setCustomEdgeDataCenters(CustomDataCenter.class);
		
		sim.launchSimulation();
	}


}
