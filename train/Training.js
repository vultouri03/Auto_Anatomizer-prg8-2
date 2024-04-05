import "https://unpkg.com/ml5@latest/dist/ml5.min.js"

//vars for the NN
const nn = ml5.neuralNetwork({ task: 'classification', debug: true,
    layers: [
        {
            type: 'dense',
            units: 64,
            activation: 'relu',
        },
        {
            type: 'dense',
            units: 64,
            activation: 'relu',
        },
        {
            type: 'dense',
            activation: 'softmax',
        } ]
})
let prediction = [];

//var for the dataset
let dataSet = []
let testData = []
let train;
let test;

//html vars
let download = document.getElementById("download")
download.addEventListener("click", downloadClickHandler)

//loads the Json training data
await fetch('../training.json')
    .then((response) => response.json())
    .then((json) => dataSet = json)

//loads the Json training data
await fetch('../test.json')
    .then((response) => response.json())
    .then((json) => testData = json)



//randomizes the data
dataSet = dataSet.toSorted(() => (Math.random() - 0.5))

//splits data into a training and test dataset
train = dataSet.slice(0, Math.floor(dataSet.length * 0.8))
test = dataSet.slice(Math.floor(dataSet.length * 0.8) + 1)

console.log(train);


//adds the data to the NN
for (let i = 0; i < train.length; i++) {
    nn.addData(train[i].pose, [train[i].label])
}

//normalizes the data and trains the nn
nn.normalizeData()
nn.train({
    epochs: 10,
    learningRate: 0.3,
}, () => finishedTraining())

//predicts the accuracy when the training is finished
async function finishedTraining(){
    download.style.hidden = false
    //randomizes testdata
    testData = test.toSorted(() => (Math.random() - 0.5))

    //predicts the testdata and pushses it to the main file
    for (let i = 0; i < testData.length; i++){
        prediction.push(await nn.classify(testData[i].pose))
    }

    //vars for the confusion matrixes data
    let both = 0;
    let bothLeft = 0;
    let bothRight = 0;
    let bothNeutral = 0

    let left = 0;
    let leftBoth = 0;
    let leftRight = 0;
    let leftNeutral = 0;

    let right = 0;
    let rightBoth = 0;
    let rightLeft = 0;
    let rightNeutral = 0;

    let neutral = 0;
    let neutralBoth = 0;
    let neutralLeft = 0;
    let neutralRight = 0;

    //vars for the confusion matrix table elements
    let both_element = document.getElementById("both_correct")
    let both_left = document.getElementById("both_left")
    let both_right = document.getElementById("both_right")
    let both_neutral = document.getElementById("both_neutral")

    let left_element = document.getElementById("left_correct")
    let left_both = document.getElementById("left_both")
    let left_right = document.getElementById("left_right")
    let left_neutral = document.getElementById("left_neutral")

    let right_element = document.getElementById("right_correct")
    let right_both = document.getElementById("right_both")
    let right_left = document.getElementById("right_left")
    let right_neutral = document.getElementById("right_neutral")

    let neutral_element = document.getElementById("neutral_correct")
    let neutral_both = document.getElementById("neutral_both")
    let neutral_left = document.getElementById("neutral_left")
    let neutral_right = document.getElementById( "neutral_right")

    //vars for the html accuracy check look
    let resultElement = document.getElementById("results")

    //vars for accuracy check
    let correctPrediction = 0;

    //checks wether the prediction is correct and increases the vars for the confusion matrix to match the mistakes the prediction made
    for(let i = 0; i < testData.length; i++){
        console.log(`i should see this ${testData[i].label} but i see this ${prediction[i][0].label}`)
        if (testData[i].label === prediction[i][0].label){
            correctPrediction ++
            switch (testData[i].label){
                case "both_up":
                    both ++;
                    break;
                case "left_up":
                    left ++;
                    break;
                case "right_up":
                    right ++;
                    break;
                case "neutral":
                    neutral ++;
            }
        } else {
            if (testData[i].label === "both_up" && prediction[i][0].label === "left_up") {
                bothLeft ++;
            } else if (testData[i].label === "both_up" && prediction[i][0].label === "right_up") {
                bothRight ++;
            } else if (testData[i].label === "both_up" && prediction[i][0].label === "neutral") {
                bothNeutral ++;
            } else if (testData[i].label === "left_up" && prediction[i][0].label === "both_up"){
                leftBoth ++;
            } else if (testData[i].label === "left_up" && prediction[i][0].label === "right_up"){
                leftRight ++
            } else if (testData[i].label === "left_up" && prediction[i][0].label === "neutral") {
                leftNeutral ++;
            } else if (testData[i].label === "right_up" && prediction[i][0].label === "both_up") {
                rightBoth ++;
            }else if (testData[i].label === "right_up" && prediction[i][0].label === "left_up") {
                rightLeft ++;
            } else if (testData[i].label === "right_up" && prediction[i][0].label === "neutral") {
                rightNeutral ++;
            } else if (testData[i].label === "neutral" && prediction[i][0].label === "both_up") {
                neutralBoth ++;
            } else if (testData[i].label === "neutral" && prediction[i][0].label === "left_up") {
                neutralLeft ++;
            } else if (testData[i].label === "neutral" && prediction[i][0].label === "right_up") {
                neutralRight ++;
            }
        }
    }
    //checks the amount of correct predictions
    console.log(both + left + right + neutral)

    //adds the confusion matrix data into the html table
    both_element.innerHTML = both.toString();
    both_left.innerHTML = bothLeft.toString();
    both_right.innerHTML = bothRight.toString();
    both_neutral.innerHTML= bothNeutral.toString()

    left_element.innerHTML = left.toString();
    left_both.innerHTML = leftBoth.toString();
    left_right.innerHTML = leftRight.toString();
    left_neutral.innerHTML = leftNeutral.toString();

    right_element.innerHTML = right.toString();
    right_both.innerHTML = rightBoth.toString();
    right_left.innerHTML = rightLeft.toString();
    right_neutral.innerHTML = rightNeutral.toString();

    neutral_element.innerHTML = neutral.toString();
    neutral_both.innerHTML = neutralBoth.toString();
    neutral_left.innerHTML = neutralLeft.toString();
    neutral_right.innerHTML = neutralRight.toString();

    let accuracy = correctPrediction / testData.length
    console.log(accuracy*100)
    resultElement.innerHTML = `total predictions :${testData.length}, correctPredictions: ${correctPrediction}, accuracy: ${accuracy*100}%`
}


//clickhandler to download the model
function downloadClickHandler() {
    nn.save("model", () => console.log("model was saved!"))
}