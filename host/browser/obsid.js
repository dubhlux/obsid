// obsid — 3D rendering foundation for Almide
//
// Built on obsid-gl (thin WebGL WASM bridge).
// Features: mesh with full TRS transform, Phong shading (diffuse + specular),
// directional + multiple point lights, fog, transparency.
//
// Usage:
//   <script src="obsid-gl.js"></script>
//   <script src="obsid.js"></script>
//   <script>Obsid.load("app.wasm", "canvas")</script>

const Obsid = {
  // ── Quality presets ───────────────────────────────
  //
  // Each entry controls: canvas backing resolution (dpr cap), MSAA on the
  // default framebuffer, texture mipmap chain + filter, anisotropic
  // filtering, and fragment-shader float precision.
  //
  // `dpr: null` means "use devicePixelRatio without a cap". Everything else
  // caps DPR at the given value so very high-DPR mobile displays don't burn
  // through fill-rate at the low/medium presets.
  QUALITIES: [
    { id: "low",    label: "Low",    dpr: 1,    msaa: false, mipmap: false, aniso: 1,  precision: "mediump" },
    { id: "medium", label: "Medium", dpr: 1.5,  msaa: true,  mipmap: false, aniso: 1,  precision: "mediump" },
    { id: "high",   label: "High",   dpr: 2,    msaa: true,  mipmap: true,  aniso: 4,  precision: "highp"   },
    { id: "max",    label: "Max",    dpr: null, msaa: true,  mipmap: true,  aniso: 16, precision: "highp"   },
  ],

  /** Resolve a quality id to a preset record. Unknown ids fall back to "max". */
  getQuality(id) {
    return this.QUALITIES.find(q => q.id === id) || this.QUALITIES[this.QUALITIES.length - 1];
  },

  /** Compute the actual backing-buffer DPR for a given quality preset. */
  dprFor(quality) {
    const native = (typeof devicePixelRatio === "number" && devicePixelRatio > 0) ? devicePixelRatio : 1;
    return quality.dpr == null ? native : Math.min(native, quality.dpr);
  },

  showError(msg) {
    let el = document.getElementById("obsid-error");
    if (!el) {
      el = document.createElement("div");
      el.id = "obsid-error";
      el.style.cssText = "position:fixed;bottom:12px;left:12px;right:12px;padding:10px;background:rgba(200,30,30,0.9);color:#fff;font:12px/1.4 monospace;border-radius:6px;z-index:9999;white-space:pre-wrap;word-break:break-all;max-height:40vh;overflow:auto";
      document.body.appendChild(el);
    }
    el.textContent = (el.textContent ? el.textContent + "\n" : "") + msg;
  },

  async load(wasmUrl, canvasEl, qualityId) {
    const quality = this.getQuality(qualityId);
    let gl, canvas;
    try {
      ({ gl, canvas } = ObsidGL.init(canvasEl, { antialias: quality.msaa }));
    } catch (e) {
      this.showError("WebGL init: " + e.message);
      throw e;
    }

    let memory;
    const state = {}, stateF = {};
    const N = (v) => Number(v);
    const B = (v) => BigInt(v ?? 0);

    // ── Mat4 ──────────────────────────────────────────
    function mat4Perspective(fov, aspect, near, far) {
      const f = 1 / Math.tan(fov * Math.PI / 360), nf = 1 / (near - far);
      return new Float32Array([f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,2*far*near*nf,0]);
    }
    function mat4LookAt(eye, center, up) {
      let fx=center[0]-eye[0],fy=center[1]-eye[1],fz=center[2]-eye[2];
      let fl=Math.hypot(fx,fy,fz); fx/=fl;fy/=fl;fz/=fl;
      let sx=fy*up[2]-fz*up[1],sy=fz*up[0]-fx*up[2],sz=fx*up[1]-fy*up[0];
      let sl=Math.hypot(sx,sy,sz); sx/=sl;sy/=sl;sz/=sl;
      let ux=sy*fz-sz*fy,uy=sz*fx-sx*fz,uz=sx*fy-sy*fx;
      return new Float32Array([
        sx,ux,-fx,0, sy,uy,-fy,0, sz,uz,-fz,0,
        -(sx*eye[0]+sy*eye[1]+sz*eye[2]),
        -(ux*eye[0]+uy*eye[1]+uz*eye[2]),
        fx*eye[0]+fy*eye[1]+fz*eye[2], 1
      ]);
    }
    function mat4Mul(a, b) {
      const o = new Float32Array(16);
      for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
        let s = 0; for (let k = 0; k < 4; k++) s += a[k*4+r]*b[c*4+k]; o[c*4+r] = s;
      }
      return o;
    }
    // T * R * S where R = Rz * Ry * Rx (column-major)
    function mat4ComposeTRS(pos, rot, scl) {
      const cx=Math.cos(rot[0]), six=Math.sin(rot[0]);
      const cy=Math.cos(rot[1]), siy=Math.sin(rot[1]);
      const cz=Math.cos(rot[2]), siz=Math.sin(rot[2]);
      const sx=scl[0], sy=scl[1], sz=scl[2];
      return new Float32Array([
        (cy*cz)*sx,                  (cy*siz)*sx,                 (-siy)*sx,   0,
        (six*siy*cz - cx*siz)*sy,    (six*siy*siz + cx*cz)*sy,    (six*cy)*sy, 0,
        (cx*siy*cz + six*siz)*sz,    (cx*siy*siz - six*cz)*sz,    (cx*cy)*sz,  0,
        pos[0], pos[1], pos[2], 1
      ]);
    }

    // ── Shader ────────────────────────────────────────
    // MAX_POINTS capped at 2 to stay well under mobile GPU
    // MAX_FRAGMENT_UNIFORM_VECTORS (minimum 16 on WebGL 1).
    const MAX_POINTS = 2;
    const VS = `
      precision highp float;
      attribute vec3 a_pos, a_norm, a_color;
      attribute vec2 a_uv;
      uniform mat4 u_vp;
      uniform mat4 u_model;
      uniform vec3 u_eye;
      varying vec3 v_norm, v_color;
      varying highp vec3 v_wpos;
      varying vec2 v_uv;
      varying highp float v_dist;
      void main() {
        vec4 world = u_model * vec4(a_pos, 1.0);
        v_wpos = world.xyz;
        v_norm = mat3(u_model) * a_norm;
        v_color = a_color;
        v_uv = a_uv;
        v_dist = length(v_wpos - u_eye);
        gl_Position = u_vp * world;
      }
    `;
    const FS = `
      precision ${quality.precision} float;
      varying vec3 v_norm, v_color;
      varying highp vec3 v_wpos;
      varying vec2 v_uv;
      varying highp float v_dist;
      uniform highp vec3 u_eye;
      uniform vec3 u_sun_color, u_sun_dir, u_ambient, u_fog_color;
      uniform vec2 u_fog_range;
      uniform float u_shininess;
      uniform vec3 u_specular;
      uniform float u_alpha;
      uniform int u_num_points;
      uniform vec3 u_point_pos[${MAX_POINTS}];
      uniform vec3 u_point_color[${MAX_POINTS}];
      uniform float u_point_range[${MAX_POINTS}];
      uniform sampler2D u_tex;
      uniform int u_has_tex;
      // sRGB → linear (gamma 2.2 approximation — cheap and close enough for
      // the 8-bit textures and user-authored colors obsid ships with).
      vec3 srgbToLinear(vec3 c) { return pow(c, vec3(2.2)); }
      // Linear → sRGB for final framebuffer output.
      vec3 linearToSrgb(vec3 c) { return pow(c, vec3(1.0 / 2.2)); }

      void main() {
        vec3 n = normalize(v_norm);
        vec3 viewDir = normalize(u_eye - v_wpos);

        // Base color: vertex color (sRGB) × texture (sRGB), both decoded to
        // linear so the lighting below composes correctly.
        vec3 baseColor = srgbToLinear(v_color);
        if (u_has_tex == 1) {
          baseColor *= srgbToLinear(texture2D(u_tex, v_uv).rgb);
        }

        // Directional (sun) — Lambert + Blinn-Phong, in linear light.
        vec3 sunColor = srgbToLinear(u_sun_color);
        vec3 sunDir = normalize(u_sun_dir);
        float sunDiff = max(dot(n, sunDir), 0.0);
        vec3 sunHalf = normalize(sunDir + viewDir);
        float sunSpec = sunDiff > 0.0 ? pow(max(dot(n, sunHalf), 0.0), u_shininess) : 0.0;
        vec3 lighting = srgbToLinear(u_ambient) + sunColor * sunDiff;
        vec3 specular = sunColor * sunSpec;

        // Point lights
        for (int i = 0; i < ${MAX_POINTS}; i++) {
          if (i >= u_num_points) break;
          vec3 toLight = u_point_pos[i] - v_wpos;
          float dist = length(toLight);
          vec3 dir = toLight / dist;
          // Smooth inverse-square-ish falloff — squared before the cutoff
          // (same as before) but now the color is linear.
          float atten = clamp(1.0 - dist / u_point_range[i], 0.0, 1.0);
          atten *= atten;
          float diff = max(dot(n, dir), 0.0);
          vec3 half_v = normalize(dir + viewDir);
          float spec = diff > 0.0 ? pow(max(dot(n, half_v), 0.0), u_shininess) : 0.0;
          vec3 pc = srgbToLinear(u_point_color[i]);
          lighting += pc * diff * atten;
          specular += pc * spec * atten;
        }

        vec3 color = baseColor * lighting + specular * u_specular;

        // Fog in linear space.
        float fog = clamp((u_fog_range.y - v_dist) / (u_fog_range.y - u_fog_range.x), 0.0, 1.0);
        color = mix(srgbToLinear(u_fog_color), color, fog);

        // Final encode back to sRGB for the framebuffer.
        gl_FragColor = vec4(linearToSrgb(color), u_alpha);
      }
    `;

    const program = ObsidGL.createProgram(gl, VS, FS);
    ObsidGL.useProgram(gl, program);
    const attr = ObsidGL.getAttribLocs(gl, program, ["a_pos", "a_norm", "a_color", "a_uv"]);
    const uni = ObsidGL.getUniformLocs(gl, program, [
      "u_vp", "u_model", "u_eye",
      "u_sun_color", "u_sun_dir", "u_ambient",
      "u_fog_color", "u_fog_range",
      "u_shininess", "u_specular", "u_alpha",
      "u_num_points",
      "u_tex", "u_has_tex",
    ]);
    // Default sampler to texture unit 0
    ObsidGL.uniform1i(gl, uni.u_tex, 0);
    // Array uniforms need per-element locations
    const uniPointPos = [], uniPointColor = [], uniPointRange = [];
    for (let i = 0; i < MAX_POINTS; i++) {
      uniPointPos.push(gl.getUniformLocation(program, `u_point_pos[${i}]`));
      uniPointColor.push(gl.getUniformLocation(program, `u_point_color[${i}]`));
      uniPointRange.push(gl.getUniformLocation(program, `u_point_range[${i}]`));
    }

    ObsidGL.enableDepthTest(gl);
    ObsidGL.enableCullFace(gl);

    // ── Mesh Storage ──────────────────────────────────
    // Vertex format: pos(3) + norm(3) + color(3) + uv(2) = 44 bytes = 11 floats
    const VERT_STRIDE = 44;
    const VERT_FLOATS = 11;

    const meshes = new Map();
    const textures = new Map();
    // Custom fragment-shader programs (obsid.create_program). Not yet wired
    // into the render loop — recorded so set_mesh_program stays harmless and a
    // future integration can pick them up. Meshes fall back to the built-in
    // lit shader until then.
    const customPrograms = new Set();
    let warnedCustomProgram = false;

    function newMesh() {
      return {
        pos: [0,0,0], rot: [0,0,0], scl: [1,1,1],
        visible: true,
        vbo: null, ibo: null, count: 0,
        shininess: 32, specular: [0,0,0], alpha: 1,
        textureId: -1,
        programId: -1, toon: null,
      };
    }

    function uploadMesh(id, vertPtr, vertCount, idxPtr, idxCount) {
      let m = meshes.get(id);
      if (!m) { m = newMesh(); meshes.set(id, m); }
      const verts = ObsidGL.viewF32(memory, vertPtr, vertCount * VERT_FLOATS);
      const indices = ObsidGL.viewU16(memory, idxPtr, idxCount);
      if (!m.vbo) { m.vbo = ObsidGL.createBuffer(gl); m.ibo = ObsidGL.createBuffer(gl); }
      ObsidGL.uploadVertexBuffer(gl, m.vbo, verts);
      ObsidGL.uploadIndexBuffer(gl, m.ibo, indices);
      m.count = idxCount;
    }

    function uploadTexture(id, dataPtr, width, height) {
      let tex = textures.get(id);
      if (tex) ObsidGL.deleteTexture(gl, tex);
      const pixels = ObsidGL.viewU8(memory, dataPtr, width * height * 4);
      tex = ObsidGL.createTexture(gl, pixels, width, height, {
        mipmap: quality.mipmap,
        aniso: quality.aniso,
      });
      textures.set(id, tex);
    }

    // Decode an encoded image (PNG/JPEG) supplied as raw bytes in WASM memory
    // and upload it as a GL texture. Decoding via createImageBitmap is async,
    // so a 1×1 white placeholder is registered immediately and swapped for the
    // real texture once it resolves — the import itself stays synchronous so
    // WASM never blocks while textures stream in.
    function uploadTextureRaw(id, dataPtr, dataLen) {
      // Copy out of WASM memory now: the bytes may be reused before decode ends.
      const encoded = ObsidGL.viewU8(memory, dataPtr, dataLen).slice();
      const placeholder = ObsidGL.createTexture(gl, new Uint8Array([255,255,255,255]), 1, 1, {});
      const prev = textures.get(id);
      if (prev) ObsidGL.deleteTexture(gl, prev);
      textures.set(id, placeholder);

      createImageBitmap(new Blob([encoded]), { premultiplyAlpha: "none", colorSpaceConversion: "none" })
        .then((bitmap) => {
          const tex = gl.createTexture();
          gl.bindTexture(gl.TEXTURE_2D, tex);
          // No Y-flip: matches the raw-pixel upload_texture path so glTF UVs
          // are interpreted identically regardless of upload route.
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, bitmap);
          const isPOT = (bitmap.width & (bitmap.width - 1)) === 0
                     && (bitmap.height & (bitmap.height - 1)) === 0;
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
          if (quality.mipmap && isPOT) {
            gl.generateMipmap(gl.TEXTURE_2D);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
          } else {
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
          }
          const wrap = isPOT ? gl.REPEAT : gl.CLAMP_TO_EDGE;
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, wrap);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, wrap);
          if (bitmap.close) bitmap.close();
          // Swap in only if this id still points at our placeholder — it may
          // have been reassigned or deleted while we were decoding.
          if (textures.get(id) === placeholder) {
            ObsidGL.deleteTexture(gl, placeholder);
            textures.set(id, tex);
          } else {
            ObsidGL.deleteTexture(gl, tex);
          }
        })
        .catch((err) => {
          console.warn(`obsid: upload_texture_raw decode failed for texture ${id}:`, err);
        });
    }

    // Re-upload only the vertex buffer of an existing mesh (indices unchanged).
    // Used for per-frame CPU skinning / morph targets: WASM recomputes the
    // vertex positions and pushes the new buffer here every frame.
    function updateMeshVerts(id, vertPtr, vertCount) {
      const m = meshes.get(id);
      if (!m || !m.vbo) return;
      const verts = ObsidGL.viewF32(memory, vertPtr, vertCount * VERT_FLOATS);
      ObsidGL.uploadVertexBuffer(gl, m.vbo, verts, gl.DYNAMIC_DRAW);
    }

    // ── Render State ──────────────────────────────────
    let cam = { fov:60, aspect:1, near:0.1, far:200, pos:[0,30,0], target:[0,0,0] };
    let sun = { color:[1,.95,.9], dir:[.5,1,.3] };
    let amb = [.15,.15,.2];
    let fog = { color:[.55,.65,.85], near:60, far:150 };
    const points = Array.from({length: MAX_POINTS}, () => ({
      pos: [0,0,0], color: [0,0,0], range: 0, active: false,
    }));

    // ── Event forwarders ──────────────────────────────
    // JS only translates browser events into raw primitive calls.
    // All state and logic lives in WASM. If the Almide module doesn't
    // export the corresponding handler, the event is silently dropped.
    let instanceRef = null;
    function call(name, ...args) {
      const fn = instanceRef && instanceRef.exports[name];
      if (fn) { try { fn(...args); } catch (e) { console.warn(name + ":", e.message); } }
    }

    // Pointer
    canvas.addEventListener("pointerdown", (e) => {
      canvas.setPointerCapture(e.pointerId);
      call("on_pointer_down", e.offsetX, e.offsetY, BigInt(e.button));
    });
    canvas.addEventListener("pointermove", (e) => {
      call("on_pointer_move", e.offsetX, e.offsetY);
    });
    canvas.addEventListener("pointerup", (e) => {
      try { canvas.releasePointerCapture(e.pointerId); } catch {}
      call("on_pointer_up", e.offsetX, e.offsetY, BigInt(e.button));
    });
    canvas.addEventListener("pointerleave", () => {
      call("on_pointer_leave");
    });
    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      call("on_wheel", e.deltaY);
    }, { passive: false });
    canvas.addEventListener("contextmenu", (e) => e.preventDefault());

    // Pinch-to-zoom — synthesise wheel delta from 2-finger touch spread
    const activeTouches = new Map();
    let pinchStartDist = 0;
    canvas.addEventListener("touchstart", (e) => {
      for (const t of e.changedTouches) {
        activeTouches.set(t.identifier, { x: t.clientX, y: t.clientY });
      }
      if (activeTouches.size === 2) {
        const [a, b] = [...activeTouches.values()];
        pinchStartDist = Math.hypot(a.x - b.x, a.y - b.y);
      }
    }, { passive: false });
    canvas.addEventListener("touchmove", (e) => {
      if (activeTouches.size === 2) {
        e.preventDefault();
        for (const t of e.changedTouches) {
          if (activeTouches.has(t.identifier)) {
            activeTouches.set(t.identifier, { x: t.clientX, y: t.clientY });
          }
        }
        const [a, b] = [...activeTouches.values()];
        const dist = Math.hypot(a.x - b.x, a.y - b.y);
        const delta = (pinchStartDist - dist) * 3; // zoom factor
        pinchStartDist = dist;
        call("on_wheel", delta);
      }
    }, { passive: false });
    canvas.addEventListener("touchend", (e) => {
      for (const t of e.changedTouches) activeTouches.delete(t.identifier);
    });
    canvas.addEventListener("touchcancel", (e) => {
      for (const t of e.changedTouches) activeTouches.delete(t.identifier);
    });

    // Keyboard (on window so canvas doesn't need focus)
    window.addEventListener("keydown", (e) => {
      call("on_key_down",
        BigInt(e.keyCode),
        BigInt(e.shiftKey ? 1 : 0),
        BigInt(e.ctrlKey ? 1 : 0),
        BigInt(e.altKey ? 1 : 0));
    });
    window.addEventListener("keyup", (e) => {
      call("on_key_up",
        BigInt(e.keyCode),
        BigInt(e.shiftKey ? 1 : 0),
        BigInt(e.ctrlKey ? 1 : 0),
        BigInt(e.altKey ? 1 : 0));
    });

    // Window lifecycle
    window.addEventListener("resize", () => {
      const dpr = Obsid.dprFor(quality);
      canvas.width = window.innerWidth * dpr;
      canvas.height = window.innerHeight * dpr;
      gl.viewport(0, 0, canvas.width, canvas.height);
      call("on_resize", BigInt(canvas.width), BigInt(canvas.height));
    });
    window.addEventListener("blur", () => call("on_blur"));
    window.addEventListener("focus", () => call("on_focus"));
    document.addEventListener("visibilitychange", () => {
      call("on_visibility_change", BigInt(document.visibilityState === "visible" ? 1 : 0));
    });

    function drawMesh(m) {
      const model = mat4ComposeTRS(m.pos, m.rot, m.scl);
      ObsidGL.uniformMatrix4fv(gl, uni.u_model, model);
      ObsidGL.uniform1f(gl, uni.u_shininess, m.shininess);
      ObsidGL.uniform3fv(gl, uni.u_specular, m.specular);
      ObsidGL.uniform1f(gl, uni.u_alpha, m.alpha);
      // Texture binding
      const tex = m.textureId >= 0 ? textures.get(m.textureId) : null;
      if (tex) {
        ObsidGL.bindTextureUnit(gl, 0, tex);
        ObsidGL.uniform1i(gl, uni.u_has_tex, 1);
      } else {
        ObsidGL.uniform1i(gl, uni.u_has_tex, 0);
      }
      ObsidGL.bindVertexAttrib(gl, m.vbo, attr.a_pos, 3, VERT_STRIDE, 0);
      ObsidGL.bindVertexAttrib(gl, m.vbo, attr.a_norm, 3, VERT_STRIDE, 12);
      ObsidGL.bindVertexAttrib(gl, m.vbo, attr.a_color, 3, VERT_STRIDE, 24);
      ObsidGL.bindVertexAttrib(gl, m.vbo, attr.a_uv, 2, VERT_STRIDE, 36);
      ObsidGL.drawElementsU16(gl, m.ibo, m.count);
    }

    function renderMeshes() {
      ObsidGL.clear(gl, fog.color[0], fog.color[1], fog.color[2]);
      const vp = mat4Mul(
        mat4Perspective(cam.fov, cam.aspect, cam.near, cam.far),
        mat4LookAt(cam.pos, cam.target, [0,1,0]),
      );
      ObsidGL.uniformMatrix4fv(gl, uni.u_vp, vp);
      ObsidGL.uniform3fv(gl, uni.u_eye, cam.pos);
      ObsidGL.uniform3fv(gl, uni.u_sun_color, sun.color);
      ObsidGL.uniform3fv(gl, uni.u_sun_dir, sun.dir);
      ObsidGL.uniform3fv(gl, uni.u_ambient, amb);
      ObsidGL.uniform3fv(gl, uni.u_fog_color, fog.color);
      ObsidGL.uniform2f(gl, uni.u_fog_range, fog.near, fog.far);

      // Upload point light uniforms
      let activeCount = 0;
      for (let i = 0; i < MAX_POINTS; i++) {
        if (points[i].active && points[i].range > 0) {
          ObsidGL.uniform3fv(gl, uniPointPos[activeCount], points[i].pos);
          ObsidGL.uniform3fv(gl, uniPointColor[activeCount], points[i].color);
          ObsidGL.uniform1f(gl, uniPointRange[activeCount], points[i].range);
          activeCount++;
        }
      }
      ObsidGL.uniform1i(gl, uni.u_num_points, activeCount);

      // Collect and sort
      const opaque = [], transparent = [];
      for (const [, m] of meshes) {
        if (!m.visible || !m.count) continue;
        (m.alpha < 1 ? transparent : opaque).push(m);
      }

      // Opaque pass
      ObsidGL.depthMask(gl, true);
      ObsidGL.disableBlend(gl);
      for (const m of opaque) drawMesh(m);

      // Transparent pass (back-to-front)
      if (transparent.length > 0) {
        transparent.sort((a, b) => {
          const da = (a.pos[0]-cam.pos[0])**2 + (a.pos[1]-cam.pos[1])**2 + (a.pos[2]-cam.pos[2])**2;
          const db = (b.pos[0]-cam.pos[0])**2 + (b.pos[1]-cam.pos[1])**2 + (b.pos[2]-cam.pos[2])**2;
          return db - da;
        });
        ObsidGL.enableBlend(gl);
        ObsidGL.depthMask(gl, false);
        for (const m of transparent) drawMesh(m);
        ObsidGL.depthMask(gl, true);
        ObsidGL.disableBlend(gl);
      }
    }

    // ── WASM Imports ──────────────────────────────────
    const imports = {
      obsid: {
        set_state(s,v){ state[N(s)]=N(v); },
        get_state(s){ return B(state[N(s)]??0); },
        set_state_f(s,v){ stateF[N(s)]=v; },
        get_state_f(s){ return stateF[N(s)]??0; },

        sin(x){return Math.sin(x);}, cos(x){return Math.cos(x);},
        sqrt(x){return Math.sqrt(x);}, abs(x){return Math.abs(x);},
        min(a,b){return Math.min(a,b);}, max(a,b){return Math.max(a,b);},
        floor(x){return Math.floor(x);}, ceil(x){return Math.ceil(x);},
        pow(x,y){return Math.pow(x,y);}, pi(){return Math.PI;},
        to_float(x){return N(x);},
        to_int(x){return B(Math.trunc(x));},

        get_aspect(){ return canvas.width/canvas.height; },

        // Mesh
        create_mesh(id){ meshes.set(N(id), newMesh()); },
        upload_mesh(id, vert_ptr, vert_count, idx_ptr, idx_count){
          uploadMesh(N(id), N(vert_ptr), N(vert_count), N(idx_ptr), N(idx_count));
        },
        set_mesh_position(id, x, y, z){
          const m = meshes.get(N(id)); if (m) m.pos = [x,y,z];
        },
        set_mesh_rotation(id, x, y, z){
          const m = meshes.get(N(id)); if (m) m.rot = [x,y,z];
        },
        set_mesh_scale(id, x, y, z){
          const m = meshes.get(N(id)); if (m) m.scl = [x,y,z];
        },
        set_mesh_material(id, shininess, sr, sg, sb){
          const m = meshes.get(N(id));
          if (m) { m.shininess = shininess; m.specular = [sr,sg,sb]; }
        },
        set_mesh_alpha(id, alpha){
          const m = meshes.get(N(id)); if (m) m.alpha = alpha;
        },
        set_mesh_visible(id, v){
          const m = meshes.get(N(id)); if (m) m.visible = N(v) !== 0;
        },
        delete_mesh(id){
          const m = meshes.get(N(id));
          if (m) {
            if (m.vbo) ObsidGL.deleteBuffer(gl, m.vbo);
            if (m.ibo) ObsidGL.deleteBuffer(gl, m.ibo);
            meshes.delete(N(id));
          }
        },

        // Texture
        upload_texture(id, data_ptr, width, height){
          uploadTexture(N(id), N(data_ptr), N(width), N(height));
        },
        set_mesh_texture(mesh_id, tex_id){
          const m = meshes.get(N(mesh_id)); if (m) m.textureId = N(tex_id);
        },
        clear_mesh_texture(mesh_id){
          const m = meshes.get(N(mesh_id)); if (m) m.textureId = -1;
        },
        delete_texture(id){
          const tex = textures.get(N(id));
          if (tex) { ObsidGL.deleteTexture(gl, tex); textures.delete(N(id)); }
        },
        upload_texture_raw(id, data_ptr, data_len){
          uploadTextureRaw(N(id), N(data_ptr), N(data_len));
        },
        update_mesh_verts(id, vert_ptr, vert_count){
          updateMeshVerts(N(id), N(vert_ptr), N(vert_count));
        },
        set_mesh_toon(mesh_id, shade_r, shade_g, shade_b, threshold){
          const m = meshes.get(N(mesh_id));
          if (m) m.toon = { shade: [shade_r, shade_g, shade_b], threshold };
        },
        create_program(id, frag_ptr, frag_len){
          // Custom fragment shaders aren't integrated into the render loop yet;
          // record the id so set_mesh_program is harmless and meshes keep using
          // the built-in lit shader.
          customPrograms.add(N(id));
          if (!warnedCustomProgram) {
            console.info("obsid: create_program — custom shaders fall back to the built-in lit shader (not yet integrated).");
            warnedCustomProgram = true;
          }
        },
        set_mesh_program(mesh_id, program_id){
          const m = meshes.get(N(mesh_id));
          if (m) m.programId = N(program_id);
        },

        // Camera & lighting
        set_camera(fov,aspect,near,far,px,py,pz,tx,ty,tz){
          cam={fov,aspect,near,far,pos:[px,py,pz],target:[tx,ty,tz]};
        },
        set_dir_light(r,g,b,dx,dy,dz){ sun={color:[r,g,b],dir:[dx,dy,dz]}; },
        set_ambient(r,g,b){ amb=[r,g,b]; },
        set_fog(r,g,b,near,far){ fog={color:[r,g,b],near,far}; },
        set_point_light(idx, x, y, z, r, g, b, range){
          const i = N(idx);
          if (i >= 0 && i < MAX_POINTS) {
            points[i] = { pos:[x,y,z], color:[r,g,b], range, active:true };
          }
        },
        clear_point_light(idx){
          const i = N(idx);
          if (i >= 0 && i < MAX_POINTS) points[i].active = false;
        },

        render(){ renderMeshes(); },
      },

      wasi_snapshot_preview1: {
        fd_write(fd,iovs,iovs_len,nwritten_ptr){
          const view=new DataView(memory.buffer);
          let written=0,output="";
          for(let i=0;i<iovs_len;i++){
            const ptr=view.getInt32(iovs+i*8,true);
            const len=view.getInt32(iovs+i*8+4,true);
            output+=new TextDecoder().decode(new Uint8Array(memory.buffer,ptr,len));
            written+=len;
          }
          if(fd===1)console.log(output.trimEnd());
          else if(fd===2)console.error(output.trimEnd());
          view.setInt32(nwritten_ptr,written,true);
          return 0;
        },
        clock_time_get(id,prec,ptr){
          new DataView(memory.buffer).setBigInt64(ptr,BigInt(Math.floor(performance.now()*1e6)),true);
          return 0;
        },
        proc_exit(c){if(c!==0)console.error(`exit(${c})`);},
        random_get(buf,len){crypto.getRandomValues(new Uint8Array(memory.buffer,buf,len));return 0;},
        path_open(){return 76;},fd_read(){return 76;},fd_close(){return 0;},
        fd_seek(){return 76;},fd_filestat_get(){return 76;},
        path_filestat_get(){return 76;},path_create_directory(){return 76;},
        path_rename(){return 76;},path_unlink_file(){return 76;},
        path_remove_directory(){return 76;},fd_prestat_get(){return 8;},
        fd_prestat_dir_name(){return 8;},fd_readdir(){return 76;},
      },
    };

    // Fetch + instantiate WASM
    let instance;
    try {
      let bytes;
      try {
        const res = await fetch(wasmUrl);
        if (!res.ok) throw new Error(`fetch ${wasmUrl}: ${res.status}`);
        bytes = await res.arrayBuffer();
      } catch (e) {
        Obsid.showError("Fetch WASM: " + e.message);
        throw e;
      }
      try {
        ({instance} = await WebAssembly.instantiate(bytes, imports));
      } catch (e) {
        Obsid.showError("Instantiate WASM:\n" + e.message);
        throw e;
      }
    } catch (e) {
      throw e;
    }
    memory = instance.exports.memory;
    instanceRef = instance;

    // Diagnostics go to console (Safari Web Inspector over USB handles mobile).
    // The red overlay stays reserved for real errors so users don't mistake
    // runtime info for a crash.
    console.log(
      "obsid: canvas=" + canvas.width + "x" + canvas.height +
      " gl=" + (gl instanceof WebGL2RenderingContext ? "webgl2" : "webgl") +
      " exports=[" + Object.keys(instance.exports).filter(n=>!n.startsWith('_')&&!n.startsWith('memory')).join(",") + "]"
    );

    if (instance.exports._start) {
      try { instance.exports._start(); }
      catch(e) {
        console.error("_start:", e);
        Obsid.showError("_start: " + e.message);
      }
    } else {
      Obsid.showError("No _start export!");
    }

    if (instance.exports.render_frame) {
      let start = null;
      let errorShown = false;
      function animate(ts) {
        if (!start) start = ts;
        try { instance.exports.render_frame((ts-start)/1000); }
        catch(e) {
          if (!errorShown) {
            console.warn("render_frame:",e.message);
            Obsid.showError("render_frame: " + e.message);
            errorShown = true;
          }
        }
        requestAnimationFrame(animate);
      }
      requestAnimationFrame(animate);
    } else {
      Obsid.showError("No render_frame export!");
    }

    return instance;
  },
};
