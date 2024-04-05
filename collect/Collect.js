import { PoseLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.0";
import kNear from "../knear.js";

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("app");
const canvasCtx = canvasElement.getContext("2d");
const drawingUtils = new DrawingUtils(canvasCtx);
const k = 3;
const machine = new kNear(k);
const button = document.getElementById("readPose");
button.addEventListener("click", buttonClickHandler);
let currentPose;
let dataSet = [];
let lineWidth = 1;
let prediction
let poseData = []
let dataArray = []
let poseSelector = document.getElementById("pose_select")
let poseLabel;


let poseLandmarker = undefined;
let lastVideoTime = -1;
let translateX = 50;
let translateY = 50;


const videoWidth =  "480px"
const videoHeight = "270px"

await createPoseLandmarker();

async function createPoseLandmarker() {
    console.log("hello")
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numPoses: 2
    })
    console.log("poselandmarker = ready")
    getVideo();
}

function getVideo() {
    const hasGetUserMedia = () => { var _a; return !!((_a = navigator.mediaDevices) === null || _a === void 0 ? void 0 : _a.getUserMedia); };

    if(hasGetUserMedia()) {
        const constraints = {
            video: {
                width: { exact: 480 },
                height: { exact: 270 }
            }
        };

        navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
            video.srcObject = stream;
            video.addEventListener("loadeddata", async () => {
                console.log(stream);
                // Once video data is loaded, you can access its width and height that you set in the constraints
                console.log(`detected webcam width: ${video.videoWidth}  Height: ${video.videoHeight}`);
                canvasElement.style.height = videoHeight; // styles need px, not just a number
                canvasElement.style.width = videoWidth;
                video.style.height = videoHeight;
                video.style.width = videoWidth;
                // Now let's start detecting the stream.
                await poseLandmarker.setOptions({ runningMode: "VIDEO"})
                await predictWebcam();
            })
        });
    }
}

async function predictWebcam() {
    let startTimeMs = performance.now();
    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;

        poseLandmarker.detectForVideo(video, startTimeMs, (result) => drawPose(result));
        // await readPrediction()
    }
        window.requestAnimationFrame(predictWebcam);

}

async function drawPose(result) {

    currentPose = result;

    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    // log de coordinaten
    // console.log(result)
    // teken de coordinaten in het canvas
    for (const landmark of result.landmarks) {
        drawingUtils.drawLandmarks(landmark, {color: "black", fillColor: "red", radius: 15});
        drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS, {color: "red", lineWidth: lineWidth});
    }

    // }

}

function buttonClickHandler() {
    const landmarks = currentPose.landmarks[0]
    poseLabel = poseSelector.value
    console.log(poseLabel)
    let output = {
        pose: [],
        label: poseLabel
    };
    if (landmarks) {
        for (let i = 0; i < landmarks.length; i++) {
            let obj = landmarks[i]
            output.pose.push(obj.x, obj.y, obj.z)
        }

    }
    dataArray.push(output)
    console.log(dataArray)
}

// async function readPrediction() {
//
//     const landmarks = currentPose.landmarks[0]
//     poseData = []
//
//
//         prediction = machine.classify(poseData)
//
//     if (prediction === "hand_open"){
//         lineWidth += 0.1;
//     }
//
//     if (prediction === "elbow_up") {
//         lineWidth -= 0.1
//         if (lineWidth < 1) {
//             lineWidth = 1
//         }
//     }
// }



