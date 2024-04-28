package pruebas;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.cloudbus.cloudsim.cloudlets.Cloudlet.Status;
import org.cloudbus.cloudsim.vms.Vm;

import com.pureedgesim.datacentersmanager.DataCenter;
import com.pureedgesim.scenariomanager.SimulationParameters;
import com.pureedgesim.simulationcore.SimLog;
import com.pureedgesim.simulationcore.SimulationManager;
import com.pureedgesim.tasksgenerator.Task;

public class MultiLayerRLManager {
	SimulationManager simulationManager;
	List<List<Integer>> orchestrationHistory;
	List<Vm> vmList;
	private boolean disableMultiLayer = false;
	
	public List<Map<String, Qrow>> vmQTableList;
	public List<List<Double>> vmAvgDataList;
	public Map<String, Qrow> Qtable = new HashMap<String, Qrow>();
	
	private double rewardSum = 0;
	private int rewardNum = 0;
	private int totalTasks = 0;
	private int askTasks = 0;
	
	// Parametro del algoritmo RL
	private double initialQvalue = 200.0;
	private double initialAskQvalue = 10;
	private double newAskQvalueDiv = 50;
	private double AskReward = 0.2;
	private double epsilon = 0.1;
	private double beta_a = 100;
	private double beta_b = 0.3;
	private double beta_c = 1;
	private double gamma = 0.3;
	private double alpha = 0.6;
		
	public MultiLayerRLManager(SimLog simLog, SimulationManager simulationManager, List<List<Integer>> orchestrationHistory, List<Vm> vmList, String algorithm) {
		this.simulationManager = simulationManager;
		this.orchestrationHistory = orchestrationHistory;
		this.vmList = vmList;

		vmQTableList = new ArrayList<>();
		vmAvgDataList = new ArrayList<>();
		for (int i = 0; i < simulationManager.getDataCentersManager().getDatacenterList().size() + 5; i++) {
			// Creating a list to store the orchestration history for each VM (virtual machine)
			vmQTableList.add(new HashMap<String, Qrow>());
			
			List<Double> listaAvg = new ArrayList<>();
			listaAvg.add(0.0); // Last Update
			listaAvg.add(0.0); // Avg cloud CPU
			listaAvg.add(0.0); // Avg edge CPU
			vmAvgDataList.add(listaAvg);
		}
		
		// Cargo los valores de las tablas Q si est� habilitado y no es el algoritmo empty
		if(!algorithm.equals("RL_MULTILAYER_EMPTY"))
			simLog.loadQTables(vmQTableList, Qtable);
		
		this.setMode(algorithm.equals("RL_MULTILAYER_DISABLED"));
	}
	
	private void updateAvgs(int localDeviceId, Task task) {
		// simulate a periodic information update system for edge devices
		// In this case they are all updated at the same time every X time

		// Estado de cloud
		double cloudCPU = simulationManager.getDataCentersManager().getDatacenterList().get(0).getResources().getAvgCpuUtilization();

		// Estado de Edge
		double edgeCPU = 0;
		for (int j = SimulationParameters.NUM_OF_CLOUD_DATACENTERS; j < SimulationParameters.NUM_OF_EDGE_DATACENTERS + SimulationParameters.NUM_OF_CLOUD_DATACENTERS; j++) {
			edgeCPU += simulationManager.getDataCentersManager().getDatacenterList().get(j).getResources().getAvgCpuUtilization();
		}
		edgeCPU /= SimulationParameters.NUM_OF_EDGE_DATACENTERS;
		
		List<Double> listaAvg = vmAvgDataList.get(localDeviceId);
		if(listaAvg.get(0) + 60 < task.getCheckTime()) {
			listaAvg.set(0, task.getCheckTime()); // Last Update
			listaAvg.set(1, cloudCPU); // Avg cloud CPU
			listaAvg.set(2, edgeCPU); // Avg edge CPU
		}
		
		
	}
	
	// Main RL offloading algorithm
	public int reinforcementLearning(String[] architecture, Task task) {	
		// *** determine the state ***
		
		// local device status
		DataCenter device = (SimulationParameters.ENABLE_ORCHESTRATORS) ? task.getOrchestrator() : task.getEdgeDevice();
		//either the orchestrator device or the edge device is assigned to the device variable. 
		
		int localDeviceId = (int) device.getId();   //extract ID of that device
		
		//Retrieves the list of virtual machines (VMs) allocated to a specific device.
		/* assumes that there is at least one host available and retrieves the list of VMs allocated to that host. 
		 * If no VMs are allocated to the device or there are no hosts, the vmListDevice list will be empty.  
		 *  */
		List<Vm> vmListDevice = device.getVmAllocationPolicy().getHostList().get(0).getVmList();
		Vm localDevice = null;
		
		// If it is a device that has computing capacity (it is not a sensor)
		if(vmListDevice.size() > 0)
			localDevice = vmListDevice.get(0); // assign the 1st VM from vmListDevice 
		
		String estado = getRLState(localDeviceId, localDevice, task);   //estado: STATE
				
		
		// *** determine the set of actions ***
		List<Qrow> acciones = getAccionesList(device, localDeviceId, localDevice, estado);
		

		// *** Exploration VS Exploitation ***
		int accion;
		double e = new Random().nextFloat();
		if(e < epsilon) { // exploration
			// random action
			accion = acciones.get(new Random().nextInt(acciones.size())).getAccion();
		} else { // exploitation: If Random No. e >= epsilon (Min Qvalue is selected)
			accion = getRLAccion(acciones);
		}
		
		// If the action to perform is to ask where to do the offloading
		String askOffloading = "false";
		if(accion == 4) {
			accion = getMultiLayerRLAccion(device, localDeviceId, localDevice, task);
			askOffloading = "true";
			askTasks++;
		}
		totalTasks++;

		// Indicates what action has been taken on this task as metadata
		task.setMetaData(new String[] { estado + "_" + accion, Integer.toString(accion), askOffloading});
		
		return accion;
	}

	public int reinforcementLearning_python(String[] architecture, Task task) {

		return 0;
	}

	private String getRLState(int localDeviceId, Vm localDevice, Task task) {
		// *** determine the original state ***
		double taskLength = task.getLength();
		double taskMaxLatency = task.getMaxLatency();
		
		// local device status
		double localCPU = 0;
		double localMIPS = 0;
		int localTaskRunning = 0;
		
		// If it is a device that has computing capacity (it is not a sensor)
		if(localDevice != null) {
			localCPU = localDevice.getCpuPercentUtilization() * 100.0; // in percent
			localMIPS = localDevice.getMips();
			localTaskRunning = orchestrationHistory.get((int) localDevice.getId()).size() - vmList.get((int) localDevice.getId()).getCloudletScheduler().getCloudletFinishedList().size() + 1;
		} else { // if it is a sensor
			
		}
		
		double avgCloudCPU = 0;
		double avgEdgeCPU = 0;

		// Average CPU usage on Cloud and Edge
		//The value at index 1 represents the average CPU usage on the cloud, and 
		//the value at index 2 represents the average CPU usage on the edge.
		if(localDeviceId != -1 ) {    // means specific device identified
			avgCloudCPU = vmAvgDataList.get(localDeviceId).get(1);
			avgEdgeCPU = vmAvgDataList.get(localDeviceId).get(2);
		} else {
			// cloud status
			avgCloudCPU = simulationManager.getDataCentersManager().getDatacenterList().get(0).getResources().getAvgCpuUtilization();

			// edge state
			for (int j = SimulationParameters.NUM_OF_CLOUD_DATACENTERS; j < SimulationParameters.NUM_OF_EDGE_DATACENTERS + SimulationParameters.NUM_OF_CLOUD_DATACENTERS; j++) {
				avgEdgeCPU += simulationManager.getDataCentersManager().getDatacenterList().get(j).getResources().getAvgCpuUtilization();
			}
			avgEdgeCPU /= SimulationParameters.NUM_OF_EDGE_DATACENTERS;
		}
		
		
		// *** I discretize the state in a finite sets *** (TODO: Fuzzification)
		String taskLengthTerm = (taskLength < 20000) ? "low" : (taskLength < 100000) ? "medium" : "high"; // MIPS required by the task 
		String taskMaxLatencyTerm = (taskMaxLatency < 6) ? "low" : (taskMaxLatency < 15) ? "medium" : "high";  // Max allowed Latency (s)
		
		
		String localCPUTerm = (localCPU < 25.0) ? "low" : (localCPU < 50) ? "medium" : (localCPU < 75) ? "busy" : "high";  // Current use of Device CPU
		String localMIPSTerm = (localMIPS < 30000) ? "low" : (localMIPS < 130000) ? "medium" : "high";     // Current use of Device MIPS

		//String edgeCPUTerm = (avgEdgeCPU < 25.0) ? "low" : (avgEdgeCPU < 75) ? "busy" : "high";
		//String cloudCPUTerm = (avgCloudCPU < 25.0) ? "low" : (avgCloudCPU < 75) ? "busy" : "high";

		//Last Avg CPU use of Fog servers
		String edgeCPUTerm = (avgEdgeCPU < 25.0) ? "low" : (avgEdgeCPU < 50) ? "medium" : (avgEdgeCPU < 75) ? "busy" : "high"; 
		
		//Last Avg CPU use of Cloud servers
		String cloudCPUTerm = (avgCloudCPU < 25.0) ? "low" : (avgCloudCPU < 50) ? "medium" : (avgCloudCPU < 75) ? "busy" : "high";

		//String estado = taskMaxLatencyTerm;
		//String estado = taskLengthTerm + "_" + taskMaxLatencyTerm + "_" + localCPUTerm + "_" + localMIPSTerm;
		String estado = cloudCPUTerm + "_"  + edgeCPUTerm + "_" + localCPUTerm + "_" + taskMaxLatencyTerm + "_" + taskLengthTerm + "_" + localMIPSTerm;
		
		return estado;    // State
	}
	
	private List<Qrow> getAccionesList(DataCenter device, int localDeviceId, Vm localDevice, String estado) {
		// *** determine the set of actions ***
		//The method initializes an empty linked list called acciones to store the available actions.
		List<Qrow> acciones = new LinkedList<Qrow>();
		
		// Local compute action (only if the device has compute capability)
		if(localDevice != null)   // local device has computing capability
			acciones.add(getQTable(localDeviceId, estado + "_0", 0)); //adds an action to acciones list by calling getQTable method

		// Mist offloading action
		// I only take it into account if I have neighbors available in my range
		if(getNumVecinos(device) > 0)    // get Number of neighbors
			acciones.add(getQTable(localDeviceId, estado + "_1", 1));
		
		// Offloading action to the edge
		acciones.add(getQTable(localDeviceId, estado + "_2", 2));

		// Offloading action to cloud
		acciones.add(getQTable(localDeviceId, estado + "_3", 3));
		
		// Ask action offloading
		if(!disableMultiLayer && localDeviceId != -1)
			acciones.add(getQTable(localDeviceId, estado + "_4", 4));
		
		return acciones;
	}
	
	private int getNumVecinos(DataCenter device) {
		int vecinos = 0;

		for (int i = 0; i < vmList.size(); i++) {
			DataCenter dcd = (DataCenter) vmList.get(i).getHost().getDatacenter();
			if(device.getId() != dcd.getId() && dcd.getType() == SimulationParameters.TYPES.EDGE_DEVICE) {
				if (device.getMobilityManager().distanceTo(dcd) < SimulationParameters.EDGE_DEVICES_RANGE) {
					vecinos++;
				}
			}
		}
		
		return vecinos;
	}

	private int getRLAccion(List<Qrow> acciones) {
		int accion = acciones.get(0).getAccion();
		double minQValue = acciones.get(0).getValue();
		
		for(int i = 1; i < acciones.size(); i++) {
			if(acciones.get(i).getValue() < minQValue) {
				minQValue = acciones.get(i).getValue();
				accion = acciones.get(i).getAccion();
			}
		}
		
		return accion;
	}
	

	private int getMultiLayerRLAccion(DataCenter device, int localDeviceId, Vm localDevice, Task task) {
		String estado = getRLState(-1, localDevice, task);
		
		// use the global QTable
		List<Qrow> acciones = getAccionesList(device, -1, localDevice, estado);
		
		int accion = getRLAccion(acciones);
		
		return accion;
	}
	

	public void reinforcementFeedback(Task task) {		
		// I calculate the reward (task execution time + waiting time)
		double executionTime = task.getActualCpuTime();
		double waitingTime = task.getExecStartTime() - task.getTime();
		double receptionTime = 0;
		if (task.getReceptionTime() != -1) // the task is offloaded
			receptionTime += task.getReceptionTime() - task.getTime();

		//double reward = executionTime + waitingTime;

		//double totalTime = task.getFinishTime() - task.getTime();
		double totalTime = task.getCheckTime() - task.getTime();
		double totalEnergy = task.getTotalCost();
		double cpuExecution = task.getVm().getCpuPercentUtilization(task.getTime());
		
		
		// Reward
		//double reward = beta_a * totalTime;
		double reward = beta_a * totalTime + beta_b * totalEnergy;
		//double reward = beta_a * totalTime + beta_b * totalEnergy + beta_c * cpuExecution;
		//double reward = (beta_a * totalTime + beta_b * totalEnergy) * cpuExecution * beta_c;
		
		if (task.getStatus() == Status.FAILED) {
			if(task.getFailureReason() == Task.Status.FAILED_DUE_TO_LATENCY) {
				//System.out.println(task.getId() + ", " + ((DataCenter)task.getVm().getHost().getDatacenter()).getType() + "; " + task.getMaxLatency() + " : " + (task.getCheckTime()-task.getTime()));
			}
			reward = beta_a*99;
		}
		
		// Action taken on that task
		String estadoTask = ((String[]) task.getMetaData())[0];
		int accion = Integer.parseInt(((String[]) task.getMetaData())[1]);
		String askOffloading = ((String[]) task.getMetaData())[2];
		
		if(askOffloading.equals("true"))
			reward *= AskReward;
		
		// local device status
		DataCenter device = (SimulationParameters.ENABLE_ORCHESTRATORS) ? task.getOrchestrator() : task.getEdgeDevice();
		int localDeviceId = (int) device.getId();
		
		List<Vm> vmListDevice = device.getVmAllocationPolicy().getHostList().get(0).getVmList();
		Vm localDevice = null;
		
		// If it is a device that has computing capacity (it is not a sensor)
		if(vmListDevice.size() > 0)
			localDevice = vmListDevice.get(0);
		
		String estadoN = getRLState(localDeviceId, localDevice, task);

		// *** Determine the set of actions and the action to be taken ***
		List<Qrow> acciones = getAccionesList(device, localDeviceId, localDevice, estadoN);
		int accionN = getRLAccion(acciones);
		
		double q = getQTable(localDeviceId, estadoN + "_" + accionN, accionN).getValue();
		
		updateQTable((int) device.getId(), estadoTask, accion, reward, q);
		
		// If it is an offloading, he decided for a higher level
		if(askOffloading.equals("true")) {
			// penalize the value of the offloading option
			String estadoTaskAsk = estadoTask.substring(0, estadoTask.length()-2) + "_4";
			double newReward = reward*(simulationManager.getSimulation().clock()/newAskQvalueDiv);
			updateQTable((int) device.getId(), estadoTaskAsk, 4, newReward, 0);

			reward *= cpuExecution * beta_c;
			
			// update the value of the global Qtable
			updateQTable(-1, estadoTask, accion, reward, q);
		}
		
		// Every time a complete task is received it checks if it is time to update the CPU usage percentages


		if(askOffloading.equals("false"))
			updateAvgs(localDeviceId, task);
	}

	private void updateQTable(int vm, String rule, int accion, double reward, double q) {
		Map<String, Qrow> Qtable = (vm == -1) ? this.Qtable : vmQTableList.get(vm);
		
		// The entry exists in the table
		if(Qtable.containsKey(rule)) {
			Qrow row = Qtable.get(rule);
			
			double QValue = row.getValue();
			row.increaseUpdatesCount();

			QValue = QValue*(1-alpha) + alpha*(reward + gamma*q); //update Q value
			row.setValue(QValue);
		} else { // There is no entry for that rule, I create it from scratch
			Qrow row = new Qrow(rule, accion, reward);
			Qtable.put(rule, row);
		}
		
		// update the local reward media counter
		if(vm != -1)
			updateAvgReward(vm, reward);
	}
	

	private Qrow getQTable(int vm, String rule, int accion) {
		Map<String, Qrow> Qtable = (vm == -1) ? this.Qtable : vmQTableList.get(vm);
		
		// The entry exists in the table
		
		//The code checks if the Qtable already contains an entry for the given rule. 
		//If it does, it retrieves and returns the corresponding Qrow object
		
		if(Qtable.containsKey(rule)) {
			return Qtable.get(rule);
		} else { // There is no entry for that rule, I create it from scratch
			//If there is no existing entry for the rule, the code creates a new Qrow object with the given rule, 
			//accion, and an initial Q-value (qValue).
			
			double qValue = (accion == 4) ? this.initialAskQvalue : this.initialQvalue;
			Qrow row = new Qrow(rule, accion, qValue);
			Qtable.put(rule, row);
			return row;
		}
		
	}
	
	private void updateAvgReward(int vm, double reward) {
		this.rewardSum += reward;
		this.rewardNum++;
	}
	
	public double getAvgReward() {
		double avgReward = this.rewardSum / this.rewardNum;
		this.rewardNum = 0;
		this.rewardSum = 0;
		return avgReward;
	}

	public List<Map<String, Qrow>> getVmQTableList() {
		return this.vmQTableList;
	}

	public int getTotalTasks() {
		return totalTasks;
	}

	public int getAskTasks() {
		return askTasks;
	}

	public void setMode(boolean disableMultiLayer) {
		this.disableMultiLayer = disableMultiLayer;
		
	}

	public void cleanQtables() {
		this.Qtable = new HashMap<String, Qrow>();
		this.vmQTableList = new ArrayList<>();
		for (int i = 0; i < simulationManager.getDataCentersManager().getDatacenterList().size() + 5; i++) {
			this.vmQTableList.add(new HashMap<String, Qrow>());
		}
		
	}
	
}
