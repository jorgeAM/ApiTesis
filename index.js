const path = require('path');
const fs = require('fs');
const fr = require('face-recognition');

//CREAR DETECTOR
const detector = fr.FaceDetector();

//CARGAR IMAGENES
const image1 = fr.loadImage('./img/l3m.jpg');
console.log('detectando rostros...');
Promise.all(detector.detectFaces(image1))
.then(l3mFaces => {
  //MOSTRAR IMAGENE
  const win = new fr.ImageWindow();
  win.setImage(fr.tileImages(l3mFaces));
  fr.hitEnterToContinue();
})
.catch(err => console.log(err));
