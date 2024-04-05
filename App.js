import {PoseLandmarker, FilesetResolver, DrawingUtils} from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.0";
import "https://unpkg.com/ml5@latest/dist/ml5.min.js"
import kNear from "./knear.js";

//html elements
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("app");
const canvasCtx = canvasElement.getContext("2d");
const startButton = document.getElementById("start");
startButton.addEventListener("click", startButtonClickHandler)
const instructions = document.getElementById("instructions");
const hideTextButton = document.getElementById("hideText");
hideTextButton.addEventListener("click", hideTextHandler)

//vars for app loop
let appStart = false
let lockedPose = [];

//vars for Mediapipe landmarking
const drawingUtils = new DrawingUtils(canvasCtx);
let poseLandmarker = undefined;
let lastVideoTime = -1;

//vars for video size
const videoWidth = "960px"
const videoHeight = "540px"

//vars for Knear model
const k = 3;
const machine = new kNear(k);
let dataSet = [];

//vars for NN
const nn = ml5.neuralNetwork({ task: 'classification', debug: true })
const modelDetails = {
    model: 'models/model.json',
    metadata: 'models/model_meta.json',
    weights: 'models/model.weights.bin'
}

nn.load(modelDetails, () => nextStep())

//vars for pose recognition
let currentPose;
let lineWidth = 1;
let prediction
let poseData = []

//reads the training data and trains our KNN
await fetch('../training.json')
    .then((response) => response.json())
    .then((json) => dataSet = json)

//train knear
// for (let i = 0; i < dataSet.length; i++) {
//     machine.learn(dataSet[i].pose, dataSet[i].label)
// }

async function nextStep() {
    await createPoseLandmarker();
}


//creates the poselandmarker and loads in the model
async function createPoseLandmarker() {
    //loads the Wasmfiles and initializes it so that it can be used for vision tasks
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm")
    //loads the model and sets it to VIDEO mode
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numPoses: 1,
        minTrackingConfidence: 0.5,
        outputSegmentationMasks: true
    })
    console.log("model is ready")

}

function getVideo() {
    //gets the available devices to stream media from
    const hasGetUserMedia = () => {
        var _a;
        return !!((_a = navigator.mediaDevices) === null || _a === void 0 ? void 0 : _a.getUserMedia);
    }
    //if the video gets read it will get set to a certain aspect ratio
    if (hasGetUserMedia()) {
        const constraints = {
            video: {
                width: {exact: 480},
                height: {exact: 270}
            }
        }
        //sets the constraints to the user media and streams the media to the video element
        navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
            video.srcObject = stream;

            video.addEventListener("loadeddata",
                async () => {
                    // Once video data is loaded, you can access its width and height that you set in the constraints
                    canvasElement.style.height = videoHeight; // styles need px, not just a number
                    canvasElement.style.width = videoWidth;
                    video.style.height = videoHeight;
                    video.style.width = videoWidth;
                    // Now let's start detecting the stream.
                    await poseLandmarker.setOptions({runningMode: "VIDEO"})
                    await predictWebcam();
                })
        });
    }
}

//function to check read the data from the webcam and predicts where the landmarks should be
async function predictWebcam() {
    //sets the start time to the current time
    let startTimeMs = performance.now();
    //creates a loop that runs every second the video time changes
    if (lastVideoTime !== video.currentTime) {
        //sets the video time to the currentTime and runs the video through the poselandmarker
        lastVideoTime = video.currentTime;
        poseLandmarker.detectForVideo(video, startTimeMs, (result) => drawPose(result));


    }
    //updates this function every frame
    window.requestAnimationFrame(predictWebcam);

}

//utilizes the drawing utils to draw the canvas
async function drawPose(result) {

    currentPose = result;

    //allows for input when a pose is a match
    await readPrediction()
    if (appStart === true) {
        console.log(appStart)
        //clears the canvas every frame so that you wont get ghost lines
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

        // teken de coordinaten in het canvas
        for (const landmark of result.landmarks) {
            drawingUtils.drawLandmarks(landmark, {color: "black", fillColor: "red", radius: 15});
            drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS, {
                color: "red",
                lineWidth: lineWidth
            });
        }
    } else  if(lockedPose !== []) {
        //clears the canvas every frame so that you wont get ghost lines
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

        // teken de coordinaten in het canvas
        for (const landmark of lockedPose.landmarks) {
            drawingUtils.drawLandmarks(landmark, {color: "black", fillColor: "red", radius: lineWidth});
            drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS, {
                color: "red",
                lineWidth: lineWidth
            });
        }
    }
}

function startButtonClickHandler() {
    startButton.classList.toggle("visually-hidden");
    hideTextButton.classList.toggle("visually-hidden");
    instructions.innerHTML =  "in order to start move both your hands up once you do this you'll have 5 seconds to strike a pose"
    getVideo()
}

function hideTextHandler() {
    instructions.classList.toggle("visually-hidden");
    hideTextButton.classList.toggle("visually-hidden")
}

//allows for input when a pose is a match
async function readPrediction() {

    const landmarks = currentPose.landmarks[0]
    poseData = []

    // reads the x, y, z data seperatly from the rest and adds those to the posedata Array
    if (landmarks) {
        for (let i = 0; i < landmarks.length; i++) {
            let obj = landmarks[i]
            poseData.push(obj.x, obj.y, obj.z)
        }
    }
    //predicts the pose for knn
    // prediction = machine.classify(poseData)

    //predicts the pose for NN
    prediction = await nn.classify(poseData)



    console.log(prediction[0].label)

    //widens the line based on said prediction
    if (prediction[0].label === "right_up" && appStart === false) {
        lineWidth += 0.1;
        if (lineWidth > 30) {
            lineWidth = 30
        }
    }

    //thins the line based on said prediction
    if (prediction[0].label === "left_up" && appStart === false) {
        lineWidth -= 0.1
        if (lineWidth < 1) {
            lineWidth = 1
        }
    }

    if (prediction[0].label === "both_up" && appStart === false){
        appStart = true
        lockPose();
    }
}

function lockPose() {
    if (instructions.classList.contains("visually-hidden")){
        instructions.classList.toggle("visually-hidden")
    }

    setTimeout(() => {
        instructions.innerHTML = "3"
    }, 1000);
    setTimeout(() => {
        instructions.innerHTML = "2"
    }, 2000);
    setTimeout(() => {
        instructions.innerHTML = "1"
    }, 3000);
    setTimeout(() => {
        instructions.innerHTML = "Pose Taken, move your right arm up to thicken the lines and your left to make them thinner again until you've got a result you are happy with"
        lockedPose = currentPose
        appStart = false;
        if (hideTextButton.classList.contains("visually-hidden")){
            hideTextButton.classList.toggle("visually-hidden")
        }
    }, 4000);
}
