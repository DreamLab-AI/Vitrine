/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "environment_renderer.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "gl_state_guard.hpp"
#include "rendering/coordinate_conventions.hpp"

#include <OpenImageIO/imageio.h>
#include <format>
#include <glad/glad.h>
#include <memory>
#include <vector>

namespace lfs::rendering {

    namespace {
        [[nodiscard]] glm::vec4 computeEnvironmentIntrinsics(const FrameView& frame_view) {
            if (frame_view.intrinsics_override.has_value() && !frame_view.orthographic) {
                const auto& intrinsics = *frame_view.intrinsics_override;
                return {intrinsics.focal_x, intrinsics.focal_y, intrinsics.center_x, intrinsics.center_y};
            }

            const auto [focal_x, focal_y] = computePixelFocalLengths(frame_view.size, frame_view.focal_length_mm);
            return {focal_x, focal_y, frame_view.size.x * 0.5f, frame_view.size.y * 0.5f};
        }
    } // namespace

    Result<void> EnvironmentRenderer::initialize() {
        if (initialized_) {
            return {};
        }

        auto shader_result = load_shader(
            "environment_equirect", "screen_quad.vert", "environment_equirect.frag", false);
        if (!shader_result) {
            return std::unexpected(shader_result.error().what());
        }
        shader_ = std::move(*shader_result);

        auto vao_result = create_vao();
        if (!vao_result) {
            return std::unexpected(vao_result.error());
        }
        vao_ = std::move(*vao_result);

        auto vbo_result = create_vbo();
        if (!vbo_result) {
            return std::unexpected(vbo_result.error());
        }
        vbo_ = std::move(*vbo_result);

        constexpr float quad_vertices[] = {
            -1.0f,
            -1.0f,
            0.0f,
            0.0f,
            1.0f,
            -1.0f,
            1.0f,
            0.0f,
            -1.0f,
            1.0f,
            0.0f,
            1.0f,
            1.0f,
            1.0f,
            1.0f,
            1.0f,
        };

        glBindVertexArray(vao_.get());
        glBindBuffer(GL_ARRAY_BUFFER, vbo_.get());
        glBufferData(GL_ARRAY_BUFFER, sizeof(quad_vertices), quad_vertices, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                              reinterpret_cast<const void*>(2 * sizeof(float)));
        glEnableVertexAttribArray(1);
        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        initialized_ = true;
        return {};
    }

    Result<void> EnvironmentRenderer::ensureTextureLoaded(const std::filesystem::path& environment_path) {
        if (environment_path == loaded_environment_path_) {
            if (texture_ready_) {
                return {};
            }
            return std::unexpected(last_load_error_);
        }

        loaded_environment_path_ = environment_path;
        texture_ready_ = false;
        last_load_error_.clear();
        environment_texture_ = Texture();

        if (environment_path.empty()) {
            last_load_error_ = "Environment map path is empty";
            return std::unexpected(last_load_error_);
        }

        if (!std::filesystem::exists(environment_path)) {
            last_load_error_ = std::format("Environment map not found: {}", environment_path.string());
            LOG_WARN("{}", last_load_error_);
            return std::unexpected(last_load_error_);
        }

        const std::string path_utf8 = lfs::core::path_to_utf8(environment_path);
        std::unique_ptr<OIIO::ImageInput> input(OIIO::ImageInput::open(path_utf8));
        if (!input) {
            last_load_error_ = std::format("Failed to open environment map {}: {}", path_utf8, OIIO::geterror());
            LOG_WARN("{}", last_load_error_);
            return std::unexpected(last_load_error_);
        }

        const auto& spec = input->spec();
        if (spec.width <= 0 || spec.height <= 0 || spec.nchannels <= 0) {
            input->close();
            last_load_error_ = std::format("Invalid environment map dimensions for {}", path_utf8);
            LOG_WARN("{}", last_load_error_);
            return std::unexpected(last_load_error_);
        }

        std::vector<float> source_pixels(
            static_cast<size_t>(spec.width) * static_cast<size_t>(spec.height) * static_cast<size_t>(spec.nchannels));
        if (!input->read_image(0, 0, 0, spec.nchannels, OIIO::TypeDesc::FLOAT, source_pixels.data())) {
            last_load_error_ = std::format("Failed to read environment map {}: {}", path_utf8, input->geterror());
            input->close();
            LOG_WARN("{}", last_load_error_);
            return std::unexpected(last_load_error_);
        }
        input->close();

        const float* upload_pixels = source_pixels.data();
        std::vector<float> rgb_pixels;
        if (spec.nchannels != 3) {
            rgb_pixels.resize(static_cast<size_t>(spec.width) * static_cast<size_t>(spec.height) * 3u);
            for (int y = 0; y < spec.height; ++y) {
                for (int x = 0; x < spec.width; ++x) {
                    const size_t src_index =
                        (static_cast<size_t>(y) * static_cast<size_t>(spec.width) + static_cast<size_t>(x)) *
                        static_cast<size_t>(spec.nchannels);
                    const size_t dst_index =
                        (static_cast<size_t>(y) * static_cast<size_t>(spec.width) + static_cast<size_t>(x)) * 3u;
                    if (spec.nchannels >= 3) {
                        rgb_pixels[dst_index + 0] = source_pixels[src_index + 0];
                        rgb_pixels[dst_index + 1] = source_pixels[src_index + 1];
                        rgb_pixels[dst_index + 2] = source_pixels[src_index + 2];
                    } else {
                        const float value = source_pixels[src_index];
                        rgb_pixels[dst_index + 0] = value;
                        rgb_pixels[dst_index + 1] = value;
                        rgb_pixels[dst_index + 2] = value;
                    }
                }
            }
            upload_pixels = rgb_pixels.data();
        }

        GLuint texture_id = 0;
        GLStateGuard state_guard;
        glGenTextures(1, &texture_id);
        if (texture_id == 0 || glGetError() != GL_NO_ERROR) {
            last_load_error_ = std::format("Failed to allocate OpenGL texture for {}", path_utf8);
            LOG_WARN("{}", last_load_error_);
            return std::unexpected(last_load_error_);
        }

        environment_texture_ = Texture(texture_id);

        glBindTexture(GL_TEXTURE_2D, environment_texture_.get());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        // `environment_equirect.frag` samples with `textureLod(..., 0)`, so generating mipmaps is wasted
        // work and can even hide issues (missing levels) in debug. Keep the texture single-level.
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        GLint previous_unpack_alignment = 4;
        glGetIntegerv(GL_UNPACK_ALIGNMENT, &previous_unpack_alignment);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, spec.width, spec.height, 0, GL_RGB, GL_FLOAT, upload_pixels);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
        glPixelStorei(GL_UNPACK_ALIGNMENT, previous_unpack_alignment);

        const GLenum texture_error = glGetError();
        if (texture_error != GL_NO_ERROR) {
            environment_texture_ = Texture();
            last_load_error_ = std::format("Failed to upload environment map {} to OpenGL", path_utf8);
            LOG_WARN("{}", last_load_error_);
            return std::unexpected(last_load_error_);
        }

        texture_ready_ = true;
        LOG_INFO("Loaded environment map {}", environment_path.string());
        return {};
    }

    Result<void> EnvironmentRenderer::render(const FrameView& frame_view,
                                             const std::filesystem::path& environment_path,
                                             const float exposure,
                                             const float rotation_degrees,
                                             const bool equirectangular_view) {
        if (!initialized_) {
            if (auto init_result = initialize(); !init_result) {
                return init_result;
            }
        }

        if (frame_view.size.x <= 0 || frame_view.size.y <= 0) {
            return std::unexpected("Environment renderer received an invalid viewport size");
        }

        if (auto texture_result = ensureTextureLoaded(environment_path); !texture_result) {
            return texture_result;
        }

        ShaderScope scope(shader_);
        if (!scope.isBound()) {
            return std::unexpected("Failed to bind environment shader");
        }

        glDisable(GL_BLEND);
        glDisable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, environment_texture_.get());
        shader_->set_uniform("u_environment", 0);
        shader_->set_uniform("u_camera_to_world", frame_view.rotation);
        shader_->set_uniform("u_viewport_size", glm::vec2(frame_view.size));
        shader_->set_uniform("u_intrinsics", computeEnvironmentIntrinsics(frame_view));
        shader_->set_uniform("u_environment_exposure", exposure);
        shader_->set_uniform("u_rotation_radians", glm::radians(rotation_degrees));
        shader_->set_uniform("u_equirectangular_view", equirectangular_view);

        glBindVertexArray(vao_.get());
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glBindVertexArray(0);
        glDepthMask(GL_TRUE);

        return {};
    }

} // namespace lfs::rendering
