package pruebas;

import com.pureedgesim.datacentersmanager.DataCenter;
import com.pureedgesim.scenariomanager.SimulationParameters;
import com.pureedgesim.simulationcore.SimLog;
import com.pureedgesim.simulationcore.SimulationManager;
import com.pureedgesim.tasksgenerator.Task;
import io.grpc.Grpc;
import io.grpc.InsecureChannelCredentials;
import io.grpc.ManagedChannel;
import org.cloudbus.cloudsim.cloudlets.Cloudlet.Status;
import org.cloudbus.cloudsim.vms.Vm;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class MultiLayerRLManager_python {
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

    private RLClient client;

    public MultiLayerRLManager_python(SimLog simLog, SimulationManager simulationManager, List<List<Integer>> orchestrationHistory, List<Vm> vmList, String algorithm) {
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


        String target = "localhost:50051";
        ManagedChannel channel = Grpc.newChannelBuilder(target, InsecureChannelCredentials.create())
                .build();

        this.client = new RLClient(channel);

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
        SimLog.println("Device" + task.getOrchestrator());


        int localDeviceId = (int) device.getId();   //extract ID of that device
        SimLog.println("LocalDevice ID" + localDeviceId );

        //Retrieves the list of virtual machines (VMs) allocated to a specific device.
        /* assumes that there is at least one host available and retrieves the list of VMs allocated to that host.
         * If no VMs are allocated to the device or there are no hosts, the vmListDevice list will be empty.
         *  */
        List<Vm> vmListDevice = device.getVmAllocationPolicy().getHostList().get(0).getVmList();
        SimLog.println("Vm List" + vmListDevice );

        for (Vm vm : vmListDevice) {
            System.out.println("VM ID: " + vm.getId());  // Print VM ID
            System.out.println("VM MIPS: " + vm.getMips());  // Print VM MIPS (Million Instructions Per Second)
            // Print other VM details as needed
            // For example, you can print VM CPU utilization, RAM, or bandwidth
        }

        Vm localDevice = null;

        // If it is a device that has computing capacity (it is not a sensor)
        if(vmListDevice.size() > 0)
            localDevice = vmListDevice.get(0); // assign the 1st VM from vmListDevice

        Map<String, Double> state = getRLState(localDeviceId, localDevice, task);   //estado: STATE
        SimLog.println("Value of RL State in Numerical Form" + state);

        String estado = getRLState_string(localDeviceId, localDevice, task);
        SimLog.println("Value of estado in String Form: " + estado); // prints fuzzy

        // *** determine the set of actions ***
        int accion = this.client.getActionFromPython(state);
        //SimLog.println("Value of Accion" + accion);


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

    private Map<String, Double> getRLState(int localDeviceId, Vm localDevice, Task task) {
        // *** determine the original state ***
        double taskLength = task.getLength();
        double taskMaxLatency = task.getMaxLatency();
        SimLog.println("Task Max Latency" + taskMaxLatency);

        // local device status
        double localCPU = 0;
        double localMIPS = 0;
        int localTaskRunning = 0;

        // If it is a device that has computing capacity (it is not a sensor)
        if(localDevice != null) {
            localCPU = localDevice.getCpuPercentUtilization() * 100.0; // in percent
            localMIPS = localDevice.getMips();
            localTaskRunning = orchestrationHistory.get((int) localDevice.getId()).size() - vmList.get((int) localDevice.getId()).getCloudletScheduler().getCloudletFinishedList().size() + 1;
            SimLog.println("Local  Task Running" + localTaskRunning);

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

        Map<String, Double> state = new HashMap<>();
       // state.put("numberOfPes", (double) task.getNumberOfPes());
        //state.put("fileSize", (double) task.getFileSize());
       // state.put("outputSize", (double) task.getOutputSize());
       // state.put("containerSize", task.getContainerSize());
        state.put("taskLength", taskLength);
        state.put("taskMaxLatency", taskMaxLatency);
        state.put("localCPU", localCPU);
        state.put("localMIPSTerm", localMIPS);
        state.put("edgeCPUTerm", avgEdgeCPU);
        state.put("cloudCPUTerm", avgCloudCPU);
        return state;
    }

    private String getRLState_string(int localDeviceId, Vm localDevice, Task task) {
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
        //String estado = cloudCPUTerm + "_"  + edgeCPUTerm + "_" + localCPUTerm + "_" + taskMaxLatencyTerm + "_" + taskLengthTerm + "_" + localMIPSTerm;
        String estado = taskLengthTerm + "_"  + taskMaxLatencyTerm + "_" +localMIPSTerm + "_"  + localCPUTerm   + "_" + edgeCPUTerm + "_" +  cloudCPUTerm;
        return estado;    // State
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

    public int getAskTasks() {
        return askTasks;
    }

    public int getTotalTasks() {
        return totalTasks;
    }

    private int getMultiLayerRLAccion(DataCenter device, int localDeviceId, Vm localDevice, Task task) {
        Map<String, Double> estado = getRLState(-1, localDevice, task);

        // use the global QTable
        int accion = this.client.getActionFromPython(estado);
        SimLog.println("Inside getMultiLayerRLAccion() Accion =" + accion);
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
//        double reward = beta_a * totalTime + beta_b * totalEnergy;
        //double reward = beta_a * totalTime + beta_b * totalEnergy + beta_c * cpuExecution;
        double reward = (beta_a * totalTime + beta_b * totalEnergy) * cpuExecution * beta_c;

        if (task.getStatus() == Status.FAILED) {
            if(task.getFailureReason() == Task.Status.FAILED_DUE_TO_LATENCY) {
                //System.out.println(task.getId() + ", " + ((DataCenter)task.getVm().getHost().getDatacenter()).getType() + "; " + task.getMaxLatency() + " : " + (task.getCheckTime()-task.getTime()));
            }
            reward = beta_a*99;
        }

        // Action taken on that task
        String estadoTask = ((String[]) task.getMetaData())[0];
//        Map<String, Double> = task
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
        if(!vmListDevice.isEmpty())
            localDevice = vmListDevice.get(0);

        Map<String, Double> estadoN = getRLState(localDeviceId, localDevice, task);
        // Access a service running on the local machine on port 50051
        int accionN = client.getActionFromPython(estadoN);

        // *** Determine the set of actions and the action to be taken ***
        client.makePythonModelLearn(estadoN, reward, false);

        // If it is an offloading, he decided for a higher level
        if(askOffloading.equals("true")) {
            // penalize the value of the offloading option
            String estadoTaskAsk = estadoTask.substring(0, estadoTask.length()-2) + "_4";
            double newReward = reward*(simulationManager.getSimulation().clock()/newAskQvalueDiv);
            updateQTable((int) device.getId(), estadoTaskAsk, accionN, newReward, 0);

            reward *= cpuExecution * beta_c;

            // update the value of the global Qtable
            client.makePythonModelLearn(estadoN, reward, false);
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


    private void updateAvgReward(int vm, double reward) {
        this.rewardSum += reward;
        this.rewardNum++;
    }

    public void setMode(boolean disableMultiLayer) {
        this.disableMultiLayer = disableMultiLayer;
    }
}
