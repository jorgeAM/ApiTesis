const fs = require('fs');
const path = require('path');
const cv = require('opencv4nodejs');
let app = require('express')();
let http = require('http').Server(app);
let io = require('socket.io')(http);
//CREANDO OBJETO CLASIFICADOR, QUE NOS PERMITIRA DETECTAR LOS ROSTORS
const classifier = new cv.CascadeClassifier(cv.HAAR_FRONTALFACE_ALT2);
//VARIABLE GETFACEIMAGE QUE BASICAMENTE IMPLEMENTA UNA FUNCION TIPO FLECHA QUE 
//PERMITE OBTENER LOS ROSTROS EN EL FRAME ENVIADO
const getFaceImage = (grayImg) => {
	const faceRects = classifier.detectMultiScale(grayImg).objects;
	if (!faceRects.length) {
		throw new Error('No se ha podido detectar ningun rostro');
	}
	return grayImg.getRegion(faceRects[0]);
};
io.on('connection', (socket) => {
    socket.on('imagen', function(imagen64){
    	console.log("Llego la imagen");
    	let base64String = imagen64.replace(/^data:([A-Za-z-+/]+);base64,/, '');
    	var bitmap = new Buffer(base64String , 'base64');
	    fs.writeFileSync("./img/image.png", bitmap);
	    console.log("Se guardo la imagen");
	    //VERIFICANDO SI EL  MODULO OPENCV ESTA CARGADO
		if (!cv.xmodules.face) {
		  throw new Error('Por favor instalar OpenCV');
		}
		//RUTA DE ARCHIVOS
		const basePath = './img';
		//CARPETA DE ROSTROS
		const imgsPath = path.resolve(basePath, 'caras');
		//NOMBRES O CLASES A MAPEAR
		const nameMappings = ['estuardo', 'rafael','jorge'];
		//LEYENDO LAS IMAGENES
		const imgFiles = fs.readdirSync(imgsPath);
	

		const trainImgs = imgFiles
		  // OBTENIENDO LA RUTA ABSOLUTA DE LOS ARCHIVOS QUE FORMAN PARTE DE NUESTRA BD DE CARAS CARAS get absolute file path
		  .map(file => res=path.resolve(imgsPath, file))
		  // LEYENDO LAS IMAGENES
		  .map(filePath =>cv.imread(filePath))
		  // CONVIRTIENDO A ESCALA DE GRISES
		  .map(img => img.bgrToGray())
		  // DETECTANDO ROSTROS
		  .map(getFaceImage)
		  // SE HACE UN REDIMENCIONAMIENTO DE LAS CARAS CON LA FINALIDAD DE TENER UNA MEJOR PRESICION AL MOMENTO DEL RECONOCIMIENTO
		  .map(faceImg => faceImg.resize(80, 80));
	 
		// CREANDO EL LABEL O TEXTO QUE INDICARA EL NOMBRE DEL ROSTRO IDENTIFICADO
		const labels = imgFiles
		  .map(file => nameMappings.findIndex(name => file.includes(name)));

		//CREANDO OBJETO QUE NOS PERMITIRA REALIZAR EL RECONOCIMIENTO USANDO EL ALGORITMO LBPH
		const lbph = new cv.LBPHFaceRecognizer();
		//ENTRENANDO A NUESTRO RECONOCEDOR CON LOS ROSTROS DEFINIDOS EN NUESTRA BASE DE DATOS
		lbph.train(trainImgs, labels);

		//LEYENDO IMAGEN EN DONDE SE REALIZARA EL RECONOCIMIENTO
		const twoFacesImg = cv.imread(path.resolve(basePath, 'image.png'));
		//REALIZANDO DETECCION DE ROSTROS
		const result = classifier.detectMultiScale(twoFacesImg.bgrToGray());
		//OPENCV TIENE LA FACILIDAD DE DETECTAR VARIOS ROSTROS CON MUCHA RAPIDEZ PERO MIENTRAS MAYOR SEAN MAS LENTO
		// SERA EL PROCESO POR LO QUE LIMITAMOS EL NUMERO DE ROSTROS DETECTADOS
		const minDetections = 10;
		//PROCEDEMOS A REALIZAR EL RECONOCIMIENTO Y EL DIBUJADO EN BASE AL NUMERO DE ROSTROS OBTENIDOS
		result.objects.forEach((faceRect, i) => {
			//REALIZANDO VALIDACION DEL NUMERO DE ROSTROS DETECTADOS
		  if (result.numDetections[i] < minDetections) {
		    return;
		  }
		  //OBTENIENDO LAS COORDENADAS DEL ROSTRO DETECTADO
		  const faceImg = twoFacesImg.getRegion(faceRect).bgrToGray();
		  //OBTENIENDO EL LABEL O NOMBRE DEL ROSTRO OBTENIDO
		  const who = nameMappings[lbph.predict(faceImg).label];
		  //DIBUJANDO CUADRO EN ROSTRO
		  const rect = cv.drawDetection(
		    twoFacesImg,
		    faceRect,
		    { color: new cv.Vec(255, 0, 0), segmentFraction: 4 }
		  );
		  //DIBUJANDO LABEL O NOMBRE DEL ROSTRO
		  const alpha = 0.4;
		  cv.drawTextBox(
		    twoFacesImg,
		    new cv.Point(rect.x, rect.y + rect.height + 10),
		    [{ text: who }],
		    alpha
		  );
		});
		//ABRIENDO VENTANA
		cv.imshowWait('result', twoFacesImg);
	});
 });
 
var port = process.env.PORT || 3001;
 
http.listen(port, function(){
   console.log('listening in http://localhost:' + port);
});

