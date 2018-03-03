const path = require('path');
const fs = require('fs');
const cv = require('opencv4nodejs')
const fr = require('face-recognition').withCv(cv);
//OBJETO CLASIFICADOR, PERMITIRA DETECTAR LOS ROSTROS USANDO EL ALGORITMO HAAR
const classifier = new cv.CascadeClassifier(cv.HAAR_FRONTALFACE_ALT2);
// OPENCV TIENE LA TENDENCIA DE DETECTAR RAPIDO ROSTROS PERO MIENTRAS MAS 
// SEAN LA PRESENCIA DE ESTOS MENOS PRECISO SERA POR LO QUE LIMITAMOS EL 
// NUMERO DE ROSTROS 
const minDetections = 5;
//CREAR DETECTOR
//ABRIENDO IMAGEN DONDE SE REALIZARA EL RECONOCIMIENTO - ESTA ES COMPATIBLE CON LA LIBRERIA OPENCV3NODE
var cvMat = cv.imread('./img/l3m.jpg');
let img=cvMat;
// REDIMENSIONANDO IMAGEN
 const minPxSize = 400000;
  if ((img.cols * img.rows) < minPxSize) {
    cvMat = img.rescale(minPxSize / (img.cols * img.rows));
  }
//CONVIRTIENDO LA IMAGEN ABIERTA AL FORMATO QUE ES LEIDO POR LA LIBRERIA FACE-RECOGNITION
const cvImg = fr.CvImage(cvMat);
//PASANDO A ESCALA DE GRISES
const grayImg = cvMat.bgrToGray();
//REALIZANDO LA DETECCIÓN DEL ROSTRO
const { objects, numDetections } = classifier.detectMultiScale(grayImg);
//OBTENIENDO ROSTROS
var faces= objects
	.filter((_, i) => minDetections <= numDetections[i])
	.map(rect => ({
	    rect,
	    face: cvMat.getRegion(rect).resize(150, 150)
	}));

//CARGANDO BASE DE DATOS
const recognizer = fr.FaceRecognizer();
const E1 = fr.loadImage('img/caras/E1.png');
const E2 = fr.loadImage('img/caras/E2.png');
const E3 = fr.loadImage('img/caras/E3.png');
const E4 = fr.loadImage('img/caras/E4.png');
const E5 = fr.loadImage('img/caras/E5.png');
const E6 = fr.loadImage('img/caras/E6.png');
const E7 = fr.loadImage('img/caras/E7.png');
const E8 = fr.loadImage('img/caras/E8.png');


const B1 = fr.loadImage('img/caras/B1.png');
const B2 = fr.loadImage('img/caras/B2.png');
const B3 = fr.loadImage('img/caras/B3.png');
const B4 = fr.loadImage('img/caras/B4.png');
const B5 = fr.loadImage('img/caras/B5.png');
const B6 = fr.loadImage('img/caras/B6.png');
const B7 = fr.loadImage('img/caras/B7.png');
const B8 = fr.loadImage('img/caras/B8.png');



const L1 = fr.loadImage('img/caras/L1.png');
const L2 = fr.loadImage('img/caras/L2.png');
const L3 = fr.loadImage('img/caras/L3.png');
const L4 = fr.loadImage('img/caras/L4.png');
const L5 = fr.loadImage('img/caras/L5.png');
const L6 = fr.loadImage('img/caras/L6.png');
const L7 = fr.loadImage('img/caras/L7.png');
const L8 = fr.loadImage('img/caras/L8.png');


//CREANDO CLASES
const LFaces = [L1,L2,L3,L4,L5,L6,L7,L8];
const EFaces = [E1,E2,E3,E4,E5,E6,E7,E8];
const JFaces = [B1,B2,B3,B4,B5,B6,B7,B8];

//EDUCANDO A NUESTRO PREDICTOR
recognizer.addFaces(LFaces, 'Rafael');
recognizer.addFaces(EFaces, 'Estuardo');
recognizer.addFaces(JFaces, 'Jorge');
//REALIZANDO PREDICCION
const prediction = recognizer.predict(cvImg);
console.log(prediction);
//EL CODIGO SIGUIENTE SOLO DESCOMENTAR CUANDO QUEREMOS ENTRENAR A NUESTRO PREDICTOR CON NUEVOS ROSTROS
//const modelState = recognizer.serialize();
//fs.writeFileSync('model.json', JSON.stringify(modelState));

//DIBUJANDO
const thickness = 2; //grosor
const color = new cv.Vec(228, 93, 1); //color del rectangulo
//RECORRIENDO TODAS LAS CARAS DETECTADAS
for (var i = 0; i<faces.length; i++) {
	//OBTENIENDO LOS DATOS DE LA PREDICCION -> NOMBRE DE ROSTRO - PORCENTAJE DE EXITO
	var text = `${prediction[i].className} (${prediction[i].distance})`;
	//DIBUJANDO RECTANGULO
	cvMat.drawRectangle(
	new cv.Point(faces[i].rect.x, faces[i].rect.y),
	new cv.Point(faces[i].rect.x + faces[i].rect.width, faces[i].rect.y + faces[i].rect.height),
		color,
		cv.LINE_4,
		thickness
	);
	//DIBUJANDO DATOS OBTENIDOS DE LA PREDICCION
	const textOffsetY = faces[i].rect.height + 20;
	cvMat.putText(
		text,
		new cv.Point(faces[i].rect.x, faces[i].rect.y + textOffsetY),
		cv.FONT_ITALIC,
		0.6,
		color,
		thickness
	);
}
//CREANDO UNA NUEVA IMAGEN COMPATIBLE CON LA LIBRERIA FACE-RECOGNITION LA CUAL YA TIENE REDIBUJADA
//LA IMAGEN TRATADA
const cvImg2 = fr.CvImage(cvMat);
const win = new fr.ImageWindow();
win.setImage(cvImg2);
fr.hitEnterToContinue();