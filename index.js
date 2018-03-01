const path = require('path');
const fs = require('fs');
const fr = require('face-recognition');

//CREAR DETECTOR
const image = fr.loadImage('./bebexito.png');
const recognizer = fr.FaceRecognizer();
const image1 = fr.loadImage('img/1.png');
const image2 = fr.loadImage('img/2.png');
const image3 = fr.loadImage('img/3.png');
const image5 = fr.loadImage('img/5.png');
const image6 = fr.loadImage('img/6.png');
const image7 = fr.loadImage('img/7.png');
const image8 = fr.loadImage('img/8.png');
const image9 = fr.loadImage('img/9.png');
const image10 = fr.loadImage('img/10.png');

const imageh1 = fr.loadImage('img/h1.png');
const imageh2 = fr.loadImage('img/h2.png');
const imageh3 = fr.loadImage('img/h3.png');
const imageh4 = fr.loadImage('img/h4.png');
const imageh5 = fr.loadImage('img/h5.png');
const imageh6 = fr.loadImage('img/h6.png');
const imageh7 = fr.loadImage('img/h7.png');
const imageh8 = fr.loadImage('img/h8.png');
const imageh9 = fr.loadImage('img/h9.png');

const bebexitoFaces = [image1, image2, image3, image5, image6, image7, image8, image9, image10];
const howarFaces    = [imageh1, imageh2, imageh3, imageh4, imageh5, imageh6, imageh7, imageh8, imageh9];
recognizer.addFaces(bebexitoFaces, 'bebexito');
recognizer.addFaces(howarFaces, 'Howar');
const predictions = recognizer.predict(image);
console.log(predictions);
console.log("-------");
const bestPrediction = recognizer.predictBest(image);
console.log(bestPrediction);
const modelState = recognizer.serialize();
fs.writeFileSync('model.json', JSON.stringify(modelState));
