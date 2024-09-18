(function () {
    // dependencies
    // - simple-noise.js
    // - https://cdnjs.cloudflare.com/ajax/libs/dat-gui/0.7.9/dat.gui.min.js

    'use strict';

    const SimplexNoise = window.SimplexNoise;
    const dat = window.dat;

    const Configs = {
        backgroundColor: '#eee9e9',
        particleNum: 1000,
        step: 5,
        base: 1000,
        zInc: 0.001,
        maxFrames: 700
    };

    let canvas, context, screenWidth, screenHeight, centerX, centerY;
    const particles = [];
    let hueBase = 0;
    let simplexNoise;
    let zoff = 0;
    let gui;
    let frameCount = 0;
    let animationId = null;

    function init() {

        canvas = document.getElementById('simplex-noise-canvas');
        if (!canvas) {
            console.error('Canvas element not found');
            return;
        }

        const targetParentId = canvas.dataset.targetParent;
        const targetParent = document.getElementById(targetParentId);
        if (!targetParent) {
            console.error(`Target parent with ID "${targetParentId}" not found`);
            return;
        }
        targetParent.appendChild(canvas);

        window.addEventListener('resize', onWindowResize);
        onWindowResize();

        for (let i = 0; i < Configs.particleNum; i++) {
            particles.push(new Particle());
            initParticle(particles[i]);
        }

        simplexNoise = new SimplexNoise();
        canvas.addEventListener('click', onCanvasClick);
        // disable GUI for now
        // setupGUI();
        update();
    }

    function setupGUI() {
        gui = new dat.GUI();
        gui.add(Configs, 'step', 1, 10);
        gui.add(Configs, 'base', 500, 3000);
        gui.add(Configs, 'zInc', 0.0001, 0.01);
        gui.close();
    }

    function onWindowResize() {
        // set canvas size to parent size
        const parent = canvas.parentElement;
        screenWidth = canvas.width = parent.offsetWidth;
        screenHeight = canvas.height = parent.offsetHeight;

        centerX = screenWidth / 2;
        centerY = screenHeight / 2;

        context = canvas.getContext('2d');
        context.lineWidth = 0.3;
        context.lineCap = context.lineJoin = 'round';
    }

    function onCanvasClick() {
        context.save();
        context.globalAlpha = 0.8;
        context.fillStyle = Configs.backgroundColor;
        context.fillRect(0, 0, screenWidth, screenHeight);
        context.restore();

        simplexNoise = new SimplexNoise();
    }

    function getNoise(x, y, z) {
        const octaves = 4;
        const fallout = 0.5;
        let amp = 1, f = 1, sum = 0;

        for (let i = 0; i < octaves; ++i) {
            amp *= fallout;
            sum += amp * (simplexNoise.noise3D(x * f, y * f, z * f) + 1) * 0.5;
            f *= 2;
        }

        return sum;
    }

    function initParticle(p) {
        p.x = p.pastX = screenWidth * Math.random();
        p.y = p.pastY = screenHeight * Math.random();
        p.color.h = hueBase + Math.atan2(centerY - p.y, centerX - p.x) * 180 / Math.PI;
        p.color.s = 1;
        p.color.l = 0.5;
        p.color.a = 0;
    }

    function update() {
        if (frameCount >= Configs.maxFrames) {
            // console.log("Animation completed after " + frameCount + " frames");
            cancelAnimationFrame(animationId);  // Cancel the animation
            return;
        }

        const { step, base } = Configs;

        particles.forEach(p => {
            p.pastX = p.x;
            p.pastY = p.y;

            const angle = Math.PI * 6 * getNoise(p.x / base * 1.75, p.y / base * 1.75, zoff);
            p.x += Math.cos(angle) * step;
            p.y += Math.sin(angle) * step;

            if (p.color.a < 1) p.color.a += 0.003;

            context.beginPath();
            context.strokeStyle = p.color.toString();
            context.moveTo(p.pastX, p.pastY);
            context.lineTo(p.x, p.y);
            context.stroke();

            if (p.x < 0 || p.x > screenWidth || p.y < 0 || p.y > screenHeight) {
                initParticle(p);
            }
        });

        hueBase += 0.1;
        zoff += Configs.zInc;

        frameCount++;
        animationId = requestAnimationFrame(update);
    }

    class HSLA {
        constructor(h = 0, s = 0, l = 0, a = 0) {
            this.h = h;
            this.s = s;
            this.l = l;
            this.a = a;
        }
        toString() {
            return `hsla(${this.h},${this.s * 100}%,${this.l * 100}%,${this.a})`;
        }
    }

    class Particle {
        constructor(x = 0, y = 0, color = new HSLA()) {
            this.x = x;
            this.y = y;
            this.color = color;
            this.pastX = this.x;
            this.pastY = this.y;
        }
    }

    init();

}());