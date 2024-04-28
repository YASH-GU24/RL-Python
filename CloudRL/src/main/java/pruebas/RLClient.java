package pruebas;

import com.pureedgesim.simulationcore.SimLog;
import io.grpc.*;
import unary.Format;
import unary.UnaryGrpc;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;

class RLClient {
    private final Logger logger = Logger.getLogger(RLClient.class.getName());

    private final UnaryGrpc.UnaryBlockingStub blockingStub;

    RLClient(Channel channel) {
        this.blockingStub = UnaryGrpc.newBlockingStub(channel);
    }

    //This Map contain  s teh state values needed to determine action

    //The method receives a Map<String, Double> named values as input,
    // which contains various features of the state required for decision-making
    public int getActionFromPython(Map<String, Double> values) {
//        logger.info("Will try to an action");
        //to convert the values map into a Format.State object, which is required by the Python service.
//The method setState is called to convert the input values map into a Format.State object.
// This object likely encapsulates the state information in a format suitable for the Python service

        Format.State state = setState(values);

        // numberOfPes
        // fileSize
        // outputSize
        // containerSize
        // maxLatency
//The method then makes a remote procedure call (RPC) to the Python service using gRPC's blockingStub.
// It sends the state information to the service and awaits a response.
        Format.Action response;
        //Format.Action object response stores the response received.

        try {
            response = blockingStub.getActionRL(state);
        } catch (StatusRuntimeException e) {
            logger.log(Level.WARNING, "RPC failed: {0}", e.getStatus());
            return -1;
        }
//        logger.info("Action Performed: " + response.getAction());
        SimLog.println("Value of Action from Python: " + response.getAction());
        return response.getAction();

    }

    public void makePythonModelLearn(Map<String, Double> new_observation, double reward, boolean done) {
//        logger.info("Python Model will try to learn now");
        Format.State new_state = setState(new_observation);

        Format.TrainModelRequest request = Format.TrainModelRequest.newBuilder()
                .setNewState(new_state)
                .setReward(reward)
                .setIsDone(done)
                .build();

        Format.Response response;
        try {
            response = blockingStub.trainModelRL(request);
        } catch (StatusRuntimeException e) {
            logger.log(Level.WARNING, "RPC failed: {0}", e.getStatus());
            return;
        }

//        logger.info(response.getMessage());
    }
//this method constructs a Format.State object based on the provided observation map,
// setting various fields based on the map's values, and returns the constructed state object.
// It's used to prepare the state information before making an RPC call to the Python service.
    private Format.State setState(Map<String, Double> observation) {
        return Format.State.newBuilder().
                setTaskLength(observation.get("taskLength")).
                setMaxLatency(observation.get("taskMaxLatency")).
                setLocalCPU(observation.get("localCPU")).
                setLocalMIPSTerm(observation.get("localMIPSTerm")).
                setEdgeCPUTerm(observation.get("edgeCPUTerm")).
                setCloudCPUTerm(observation.get("cloudCPUTerm")).
                build();
                //setTaskMaxLatency(observation.get("TaskMaxLength")).
                //setNumberOfPes(observation.get("numberOfPes")).
                //setFileSize(observation.get("fileSize")).
                //setOutputSize(observation.get("outputSize")).
                //setContainerSize(observation.get("containerSize")).


    }

    public static void main(String[] args) throws InterruptedException {
        String user = "world";
        // Access a service running on the local machine on port 50051
        String target = "localhost:50051";

        ManagedChannel channel = Grpc.newChannelBuilder(target, InsecureChannelCredentials.create())
                .build();

        Map<String, Double> state = new HashMap<>();
        state.put("taskLength", 0.0);
        state.put("taskMaxLatency", 0.0);
        state.put("localCPU", 0.0);
        state.put("localMIPSTerm", 0.0);
        state.put("edgeCPUTerm", 0.0);
        state.put("cloudCPUTerm", 0.0);

        Map<String, Double> new_state = new HashMap<>();
        new_state.put("taskLength", 0.0);
        new_state.put("taskMaxLatency", 0.0);
        new_state.put("localCPU", 0.0);
        new_state.put("localMIPSTerm", 0.0);
        new_state.put("edgeCPUTerm", 0.0);
        new_state.put("cloudCPUTerm", 0.0);

        try {
            RLClient client = new RLClient(channel);
            client.getActionFromPython(state);
            client.makePythonModelLearn(new_state, 0.0, false);
            client.makePythonModelLearn(new_state, 0.0, false);
        } finally {
            // ManagedChannels use resources like threads and TCP connections. To prevent leaking these
            // resources the channel should be shut down when it will no longer be used. If it may be used
            // again leave it running.
            channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
        }
    }
}