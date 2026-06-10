// almide/wasm-webgl — JS host for WebGL WASM bindings
//
// Usage:
//   const instance = await AlmideWebGL.load("app.wasm", "canvas");
//
// Note: Almide Int = WASM i64 = JS BigInt. All Int params/returns are converted.

const AlmideWebGL = {
  async load(wasmUrl, canvasEl) {
    const canvas = typeof canvasEl === "string"
      ? document.getElementById(canvasEl)
      : canvasEl;
    if (!canvas) throw new Error(`Canvas not found: ${canvasEl}`);
    const gl = canvas.getContext("webgl");
    if (!gl) throw new Error("WebGL not supported");

    let memory;
    const handles = [null]; // index 0 = null handle
    const state = {};

    // i64 ↔ JS Number conversion helpers
    const N = (big) => Number(big);  // BigInt → Number
    const B = (num) => BigInt(num ?? 0);  // Number → BigInt

    function addHandle(obj) {
      if (obj === null) return B(0);
      handles.push(obj);
      return B(handles.length - 1);
    }
    const texContent = new WeakSet();
      function getHandle(id) {
      const idx = (typeof id === "bigint") ? N(id) : id;
      return idx > 0 ? handles[idx] : null;
    }

    // Almide container ABI (compiler ≥ 0.26): String / List are laid out as
    //   [len: i32][cap: i32][payload...]
    // so the payload (UTF-8 bytes / elements) starts 8 bytes in, NOT 4.
    const HEADER = 8;

    function readString(ptr) {
      const p = N(ptr);
      const view = new DataView(memory.buffer);
      const len = view.getInt32(p, true);
      return new TextDecoder().decode(new Uint8Array(memory.buffer, p + HEADER, len));
    }

    function readListF64(ptr) {
      const p = N(ptr);
      const view = new DataView(memory.buffer);
      const len = view.getInt32(p, true);
      const arr = new Float32Array(len);
      for (let i = 0; i < len; i++) {
        arr[i] = view.getFloat64(p + HEADER + i * 8, true);
      }
      return arr;
    }

    function readListI64asU16(ptr) {
      const p = N(ptr);
      const view = new DataView(memory.buffer);
      const len = view.getInt32(p, true);
      const arr = new Uint16Array(len);
      for (let i = 0; i < len; i++) {
        arr[i] = Number(view.getBigInt64(p + HEADER + i * 8, true));
      }
      return arr;
    }

    const imports = {
      webgl: {
        // State storage
        set_state(slot, value) { state[N(slot)] = N(value); },
        get_state(slot) { return B(state[N(slot)] ?? 0); },

        // Context
        viewport(x, y, w, h) { gl.viewport(N(x), N(y), N(w), N(h)); },
        clear_color(r, g, b, a) { gl.clearColor(r, g, b, a); },
        clear(mask) { gl.clear(N(mask)); },
        enable(cap) { gl.enable(N(cap)); },
        disable(cap) { gl.disable(N(cap)); },
        depth_func(f) { gl.depthFunc(N(f)); },
        blend_func(s, d) { gl.blendFunc(N(s), N(d)); },
        cull_face(m) { gl.cullFace(N(m)); },

        // Shaders
        create_shader(type) { return addHandle(gl.createShader(N(type))); },
        shader_source(sh, srcPtr) { gl.shaderSource(getHandle(sh), readString(srcPtr)); },
        compile_shader(sh) {
          const shader = getHandle(sh);
          gl.compileShader(shader);
          if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            console.error("Shader error:", gl.getShaderInfoLog(shader));
            return B(0);
          }
          return B(1);
        },
        delete_shader(sh) { gl.deleteShader(getHandle(sh)); },

        // Programs
        create_program() { return addHandle(gl.createProgram()); },
        attach_shader(p, sh) { gl.attachShader(getHandle(p), getHandle(sh)); },
        link_program(p) {
          const prog = getHandle(p);
          gl.linkProgram(prog);
          if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
            console.error("Link error:", gl.getProgramInfoLog(prog));
            return B(0);
          }
          return B(1);
        },
        use_program(p) { gl.useProgram(getHandle(p)); },
        delete_program(p) { gl.deleteProgram(getHandle(p)); },

        // Attributes
        get_attrib_location(p, namePtr) {
          return B(gl.getAttribLocation(getHandle(p), readString(namePtr)));
        },
        enable_vertex_attrib_array(i) { gl.enableVertexAttribArray(N(i)); },
        vertex_attrib_pointer(index, size, type, normalized, stride, offset) {
          gl.vertexAttribPointer(N(index), N(size), N(type), !!N(normalized), N(stride), N(offset));
        },

        // Uniforms
        get_uniform_location(p, namePtr) {
          return addHandle(gl.getUniformLocation(getHandle(p), readString(namePtr)));
        },
        uniform1i(loc, v) { gl.uniform1i(getHandle(loc), N(v)); },
        uniform1f(loc, v) { gl.uniform1f(getHandle(loc), v); },
        uniform2f(loc, x, y) { gl.uniform2f(getHandle(loc), x, y); },
        uniform3f(loc, x, y, z) { gl.uniform3f(getHandle(loc), x, y, z); },
        uniform4f(loc, x, y, z, w) { gl.uniform4f(getHandle(loc), x, y, z, w); },
        uniform_matrix4fv(loc, dataPtr) {
          gl.uniformMatrix4fv(getHandle(loc), false, readListF64(dataPtr));
        },

        // Buffers
        create_buffer() { return addHandle(gl.createBuffer()); },
        bind_buffer(target, buf) { gl.bindBuffer(N(target), getHandle(buf)); },
        buffer_data_f32(target, dataPtr, usage) {
          gl.bufferData(N(target), readListF64(dataPtr), N(usage));
        },
        buffer_data_u16(target, dataPtr, usage) {
          gl.bufferData(N(target), readListI64asU16(dataPtr), N(usage));
        },
        // Zero-copy uploads from a raw Bytes payload (bytes.data_ptr points at
        // the raw payload, with NO [len][cap] header). byte_len is in bytes.
        // Preferred for large meshes (VRM) over the List[Float] variants.
        buffer_data_f32_ptr(target, ptr, byte_len, usage) {
          const view = new Float32Array(memory.buffer, N(ptr), N(byte_len) / 4);
          gl.bufferData(N(target), view, N(usage));
        },
        buffer_data_u16_ptr(target, ptr, byte_len, usage) {
          const view = new Uint16Array(memory.buffer, N(ptr), N(byte_len) / 2);
          gl.bufferData(N(target), view, N(usage));
        },

        // Textures
        create_texture() { return addHandle(gl.createTexture()); },
        bind_texture(target, tex) { gl.bindTexture(N(target), getHandle(tex)); },
        tex_parameteri(target, pname, param) { gl.texParameteri(N(target), N(pname), N(param)); },
        active_texture(unit) { gl.activeTexture(N(unit)); },
        // Decode an encoded image (PNG/JPEG) from a raw Bytes payload and upload
        // it to texture handle `tex`. Async: a 1×1 white placeholder is set now
        // and replaced once decode resolves, so the import stays synchronous and
        // WASM never blocks. Sets sensible filters/wrap (mipmaps when POT).
        tex_image_2d_encoded(tex, ptr, byte_len) {
          const handle = getHandle(tex);
          const encoded = new Uint8Array(memory.buffer, N(ptr), N(byte_len)).slice();
          // Placeholder only on FIRST upload; re-uploads keep the old pixels
          // until the new bitmap is decoded (no white flash on live recolor).
          if (!texContent.has(handle)) {
            gl.bindTexture(gl.TEXTURE_2D, handle);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
                          new Uint8Array([255, 255, 255, 255]));
          }
          createImageBitmap(new Blob([encoded]), { premultiplyAlpha: "none", colorSpaceConversion: "none" })
            .then((bmp) => {
              texContent.add(handle);
              gl.bindTexture(gl.TEXTURE_2D, handle);
              gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, bmp);
              const isPOT = (bmp.width & (bmp.width - 1)) === 0 && (bmp.height & (bmp.height - 1)) === 0;
              gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
              if (isPOT) {
                gl.generateMipmap(gl.TEXTURE_2D);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
              } else {
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
              }
              const wrap = isPOT ? gl.REPEAT : gl.CLAMP_TO_EDGE;
              gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, wrap);
              gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, wrap);
              if (bmp.close) bmp.close();
            })
            .catch((e) => console.warn("tex_image_2d_encoded decode failed:", e));
        },

        // Drawing
        draw_arrays(mode, first, count) { gl.drawArrays(N(mode), N(first), N(count)); },
        draw_elements(mode, count, type, offset) { gl.drawElements(N(mode), N(count), N(type), N(offset)); },
      },

      wasi_snapshot_preview1: {
        fd_write(fd, iovs, iovs_len, nwritten_ptr) {
          const view = new DataView(memory.buffer);
          let written = 0;
          let output = "";
          for (let i = 0; i < iovs_len; i++) {
            const ptr = view.getInt32(iovs + i * 8, true);
            const len = view.getInt32(iovs + i * 8 + 4, true);
            output += new TextDecoder().decode(new Uint8Array(memory.buffer, ptr, len));
            written += len;
          }
          if (fd === 1) console.log(output.trimEnd());
          else if (fd === 2) console.error(output.trimEnd());
          view.setInt32(nwritten_ptr, written, true);
          return 0;
        },
        clock_time_get(id, precision, time_ptr) {
          const view = new DataView(memory.buffer);
          view.setBigInt64(time_ptr, BigInt(Math.floor(performance.now() * 1e6)), true);
          return 0;
        },
        proc_exit(code) { if (code !== 0) console.error(`exit(${code})`); },
        random_get(buf, len) {
          crypto.getRandomValues(new Uint8Array(memory.buffer, buf, len));
          return 0;
        },
        path_open() { return 76; },
        fd_read() { return 76; },
        fd_close() { return 0; },
        fd_seek() { return 76; },
        fd_filestat_get() { return 76; },
        path_filestat_get() { return 76; },
        path_create_directory() { return 76; },
        path_rename() { return 76; },
        path_unlink_file() { return 76; },
        path_remove_directory() { return 76; },
        fd_prestat_get() { return 8; },
        fd_prestat_dir_name() { return 8; },
        fd_readdir() { return 76; },
      },
    };

    const response = await fetch(wasmUrl);
    const bytes = await response.arrayBuffer();
    const { instance } = await WebAssembly.instantiate(bytes, imports);
    memory = instance.exports.memory;

    // Run _start (setup)
    if (instance.exports._start) {
      instance.exports._start();
    }

    // Animation loop — calls exported render_frame(time: Float)
    if (instance.exports.render_frame) {
      let start = null;
      function animate(timestamp) {
        if (!start) start = timestamp;
        const time = (timestamp - start) / 1000.0;
        instance.exports.render_frame(time);
        requestAnimationFrame(animate);
      }
      requestAnimationFrame(animate);
    }

    return instance;
  },
};
