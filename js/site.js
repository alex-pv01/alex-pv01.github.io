var particlesJSConfig = {
  "particles": {
    "number": {
      "value": 80,
      "density": {
        "enable": true,
        "value_area": 800
      }
    },
    "color": {
      "value": "#f4f4f4"
    },
    "shape": {
      "type": "circle",
      "stroke": {
        "width": 0,
        "color": "#f4f4f4"
      },
      "polygon": {
        "nb_sides": 5
      },
    },
    "opacity": {
      "value": 1,
      "random": false
    },
    "size": {
      "value": 15,
      "random": true,
      "anim": {
        "enable": false,
        "speed": 20,
        "size_min": 10,
        "sync": false
      }
    },
    "line_linked": {
      "enable": true,
      "distance": 300,
      "color": "#f4f4f4",
      "opacity": 1,
      "width": 2
    },
    "move": {
      "enable": true,
      "speed": 2,
      "direction": "none",
      "random": false,
      "straight": false,
      "out_mode": "out",
      "bounce": false,
      "attract": {
        "enable": false,
        "rotateX": 800,
        "rotateY": 1200
      }
    }
  },
  "retina_detect": true
};
window.particlesJS('particles-js', particlesJSConfig);
