/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/filesystem_utils.hpp"
#include "io/formats/colmap.hpp"
#include "io/loaders/colmap_loader.hpp"

#include <atomic>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <sstream>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;

namespace {

    class ColmapImageLayoutTest : public ::testing::Test {
    protected:
        void SetUp() override {
            temp_dir_ = fs::temp_directory_path() / "lfs_colmap_image_layout_test";
            std::error_code ec;
            fs::remove_all(temp_dir_, ec);
            fs::create_directories(temp_dir_);
        }

        void TearDown() override {
            std::error_code ec;
            fs::remove_all(temp_dir_, ec);
        }

        void write_text_file(const fs::path& path, const std::string& contents) {
            fs::create_directories(path.parent_path());
            std::ofstream out(path, std::ios::binary);
            ASSERT_TRUE(out.is_open()) << "Failed to open " << path;
            out << contents;
            out.close();
            ASSERT_TRUE(out.good()) << "Failed to write " << path;
        }

        void write_png(const fs::path& path) {
            static const std::vector<unsigned char> PNG_1X1 = {
                0x89,
                0x50,
                0x4E,
                0x47,
                0x0D,
                0x0A,
                0x1A,
                0x0A,
                0x00,
                0x00,
                0x00,
                0x0D,
                0x49,
                0x48,
                0x44,
                0x52,
                0x00,
                0x00,
                0x00,
                0x01,
                0x00,
                0x00,
                0x00,
                0x01,
                0x08,
                0x06,
                0x00,
                0x00,
                0x00,
                0x1F,
                0x15,
                0xC4,
                0x89,
                0x00,
                0x00,
                0x00,
                0x0D,
                0x49,
                0x44,
                0x41,
                0x54,
                0x78,
                0x9C,
                0x63,
                0x00,
                0x01,
                0x00,
                0x00,
                0x05,
                0x00,
                0x01,
                0x0D,
                0x0A,
                0x2D,
                0xB4,
                0x00,
                0x00,
                0x00,
                0x00,
                0x49,
                0x45,
                0x4E,
                0x44,
                0xAE,
                0x42,
                0x60,
                0x82,
            };

            fs::create_directories(path.parent_path());
            std::ofstream out(path, std::ios::binary);
            ASSERT_TRUE(out.is_open()) << "Failed to open " << path;
            out.write(reinterpret_cast<const char*>(PNG_1X1.data()),
                      static_cast<std::streamsize>(PNG_1X1.size()));
            out.close();
            ASSERT_TRUE(out.good()) << "Failed to write " << path;
        }

        void write_minimal_colmap_text_dataset(const fs::path& dataset_dir,
                                               const std::vector<std::string>& image_names) {
            write_text_file(dataset_dir / "cameras.txt",
                            "1 PINHOLE 1 1 1 1 0.5 0.5\n");

            std::ostringstream images;
            for (size_t i = 0; i < image_names.size(); ++i) {
                images << (i + 1) << " 1 0 0 0 0 0 0 1 " << image_names[i] << "\n";
            }
            write_text_file(dataset_dir / "images.txt", images.str());
        }

        void write_minimal_colmap_text_dataset(const fs::path& dataset_dir,
                                               const std::string& image_name) {
            write_minimal_colmap_text_dataset(dataset_dir, std::vector<std::string>{image_name});
        }

        void write_ascii_double_ply(const fs::path& path, const size_t vertex_count) {
            fs::create_directories(path.parent_path());
            std::ofstream out(path, std::ios::binary);
            ASSERT_TRUE(out.is_open()) << "Failed to open " << path;
            out << "ply\n";
            out << "format ascii 1.0\n";
            out << "element vertex " << vertex_count << "\n";
            out << "property double x\n";
            out << "property double y\n";
            out << "property double z\n";
            out << "end_header\n";
            for (size_t i = 0; i < vertex_count; ++i) {
                out << static_cast<double>(i) << ' '
                    << static_cast<double>(i) * 0.5 << ' '
                    << static_cast<double>(i) * -0.25 << '\n';
            }
            out.close();
            ASSERT_TRUE(out.good()) << "Failed to write " << path;
        }

        fs::path temp_dir_;
    };

} // namespace

TEST_F(ColmapImageLayoutTest, ResolvesNestedImagesByBasename) {
    const fs::path dataset_dir = temp_dir_ / "dataset";
    const fs::path nested_image =
        dataset_dir / "images" / "Photogrammetry Sekal pipes" / "frame_0001.png";
    const fs::path nested_mask =
        dataset_dir / "masks" / "Photogrammetry Sekal pipes" / "frame_0001.png";

    write_minimal_colmap_text_dataset(dataset_dir, "frame_0001.png");
    write_png(nested_image);
    write_png(nested_mask);

    auto result =
        lfs::io::read_colmap_cameras_and_images_text(dataset_dir, "images");
    ASSERT_TRUE(result.has_value()) << result.error().format();

    auto& [cameras, scene_center] = *result;
    (void)scene_center;

    ASSERT_EQ(cameras.size(), 1u);
    EXPECT_EQ(cameras[0]->image_name(), "frame_0001.png");
    EXPECT_TRUE(fs::equivalent(cameras[0]->image_path(), nested_image));
    EXPECT_TRUE(fs::equivalent(cameras[0]->mask_path(), nested_mask));
}

TEST_F(ColmapImageLayoutTest, ResolvesDuplicateNestedImagesAndMasksByRelativePath) {
    const fs::path dataset_dir = temp_dir_ / "dataset";
    const fs::path image_a = dataset_dir / "images" / "img1" / "frame_0001.png";
    const fs::path image_b = dataset_dir / "images" / "img2" / "frame_0001.png";
    const fs::path mask_a = dataset_dir / "masks" / "img1" / "frame_0001.png";
    const fs::path mask_b = dataset_dir / "masks" / "img2" / "frame_0001.png";

    write_minimal_colmap_text_dataset(
        dataset_dir,
        std::vector<std::string>{"img1/frame_0001.png", "img2/frame_0001.png"});
    write_png(image_a);
    write_png(image_b);
    write_png(mask_a);
    write_png(mask_b);

    auto result =
        lfs::io::read_colmap_cameras_and_images_text(dataset_dir, "images");
    ASSERT_TRUE(result.has_value()) << result.error().format();

    auto& [cameras, scene_center] = *result;
    (void)scene_center;

    ASSERT_EQ(cameras.size(), 2u);
    EXPECT_EQ(cameras[0]->image_name(), "img1/frame_0001.png");
    EXPECT_EQ(cameras[1]->image_name(), "img2/frame_0001.png");
    EXPECT_TRUE(fs::equivalent(cameras[0]->image_path(), image_a));
    EXPECT_TRUE(fs::equivalent(cameras[1]->image_path(), image_b));
    EXPECT_TRUE(fs::equivalent(cameras[0]->mask_path(), mask_a));
    EXPECT_TRUE(fs::equivalent(cameras[1]->mask_path(), mask_b));
}

TEST_F(ColmapImageLayoutTest, FailsWhenDuplicateNestedImagesAreReferencedByBasename) {
    const fs::path dataset_dir = temp_dir_ / "dataset";
    const fs::path image_a = dataset_dir / "images" / "img1" / "frame_0001.png";
    const fs::path image_b = dataset_dir / "images" / "img2" / "frame_0001.png";

    write_minimal_colmap_text_dataset(
        dataset_dir,
        std::vector<std::string>{"frame_0001.png", "frame_0001.png"});
    write_png(image_a);
    write_png(image_b);

    auto result =
        lfs::io::read_colmap_cameras_and_images_text(dataset_dir, "images");
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, lfs::io::ErrorCode::INVALID_DATASET);
    EXPECT_NE(result.error().message.find("basename only"), std::string::npos);
    EXPECT_NE(result.error().message.find("relative image path"), std::string::npos);
}

TEST_F(ColmapImageLayoutTest, ValidationFailsWhenDuplicateBasenameWasCollapsedInMetadata) {
    const fs::path dataset_dir = temp_dir_ / "dataset";

    write_minimal_colmap_text_dataset(dataset_dir, "frame_0001.png");
    write_png(dataset_dir / "images" / "img1" / "frame_0001.png");
    write_png(dataset_dir / "images" / "img2" / "frame_0001.png");

    auto result = lfs::io::validate_colmap_dataset_layout(dataset_dir, "images");
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, lfs::io::ErrorCode::INVALID_DATASET);
    EXPECT_NE(result.error().message.find("basename only"), std::string::npos);
    EXPECT_NE(result.error().message.find("Metadata contains 1 record"), std::string::npos);
    EXPECT_NE(result.error().message.find("flattened or dropped"), std::string::npos);
}

TEST_F(ColmapImageLayoutTest, ValidationFailsWhenMasksDoNotMirrorRelativeImageLayout) {
    const fs::path dataset_dir = temp_dir_ / "dataset";

    write_minimal_colmap_text_dataset(
        dataset_dir,
        std::vector<std::string>{"img1/frame_0001.png", "img2/frame_0001.png"});
    write_png(dataset_dir / "images" / "img1" / "frame_0001.png");
    write_png(dataset_dir / "images" / "img2" / "frame_0001.png");
    write_png(dataset_dir / "masks" / "cam_a" / "frame_0001.png");
    write_png(dataset_dir / "masks" / "cam_b" / "frame_0001.png");

    auto result = lfs::io::validate_colmap_dataset_layout(dataset_dir, "images");
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, lfs::io::ErrorCode::INVALID_DATASET);
    EXPECT_NE(result.error().message.find("mask"), std::string::npos);
    EXPECT_NE(result.error().message.find("masks/img1/frame_0001.png"), std::string::npos);
}

TEST_F(ColmapImageLayoutTest, FailsEarlyWhenReferencedImageIsMissing) {
    const fs::path dataset_dir = temp_dir_ / "dataset";

    write_minimal_colmap_text_dataset(dataset_dir, "missing_frame.png");

    auto result =
        lfs::io::read_colmap_cameras_and_images_text(dataset_dir, "images");
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, lfs::io::ErrorCode::PATH_NOT_FOUND);
}

TEST_F(ColmapImageLayoutTest, ValidateOnlyRunsColmapPreflight) {
    const fs::path dataset_dir = temp_dir_ / "dataset";

    write_minimal_colmap_text_dataset(dataset_dir, "frame_0001.png");
    write_png(dataset_dir / "images" / "img1" / "frame_0001.png");
    write_png(dataset_dir / "images" / "img2" / "frame_0001.png");

    lfs::io::ColmapLoader loader;
    auto result = loader.load(dataset_dir, {.validate_only = true});

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, lfs::io::ErrorCode::INVALID_DATASET);
    EXPECT_NE(result.error().message.find("basename only"), std::string::npos);
}

TEST_F(ColmapImageLayoutTest, ValidationCanBeCancelledDuringFilesystemScan) {
    const fs::path dataset_dir = temp_dir_ / "dataset";

    write_minimal_colmap_text_dataset(dataset_dir, "frame_0001.png");
    write_png(dataset_dir / "images" / "frame_0001.png");

    int cancel_checks = 0;
    auto result = lfs::io::validate_colmap_dataset_layout(dataset_dir, "images", {
                                                                                     .cancel_requested = [&cancel_checks]() {
                                                                                         return ++cancel_checks >= 1;
                                                                                     },
                                                                                 });

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, lfs::io::ErrorCode::CANCELLED);
}

TEST_F(ColmapImageLayoutTest, LoadCanBeCancelledDuringMetadataParse) {
    const fs::path dataset_dir = temp_dir_ / "dataset";

    write_minimal_colmap_text_dataset(dataset_dir, "frame_0001.png");
    write_png(dataset_dir / "images" / "frame_0001.png");

    int cancel_checks = 0;
    lfs::io::ColmapLoader loader;
    auto result = loader.load(dataset_dir, {
                                               .cancel_requested = [&cancel_checks]() {
                                                   return ++cancel_checks >= 2;
                                               },
                                           });

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, lfs::io::ErrorCode::CANCELLED);
}

TEST_F(ColmapImageLayoutTest, LoadCancelsInsteadOfFallingBackFromCustomPointCloudPly) {
    const fs::path dataset_dir = temp_dir_ / "dataset";

    write_minimal_colmap_text_dataset(dataset_dir, "frame_0001.png");
    write_png(dataset_dir / "images" / "frame_0001.png");
    write_ascii_double_ply(dataset_dir / "points3D.ply", 300000);

    std::atomic<bool> cancel_requested{false};
    std::atomic<int> cancel_checks{0};
    cancel_requested.store(true, std::memory_order_relaxed);

    lfs::io::ColmapLoader loader;
    auto result = loader.load(dataset_dir, {
                                               .cancel_requested = [&cancel_requested, &cancel_checks]() {
                                                   const int check_count = cancel_checks.fetch_add(1, std::memory_order_relaxed) + 1;
                                                   return cancel_requested.load(std::memory_order_relaxed) && check_count >= 50;
                                               },
                                           });

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, lfs::io::ErrorCode::CANCELLED);
}

TEST_F(ColmapImageLayoutTest, DetectDatasetInfoCountsNestedImagesAndMasks) {
    const fs::path dataset_dir = temp_dir_ / "dataset";

    write_png(dataset_dir / "images" / "img1" / "frame_0001.png");
    write_png(dataset_dir / "images" / "img2" / "frame_0002.png");
    write_png(dataset_dir / "masks" / "img1" / "frame_0001.png");
    write_png(dataset_dir / "masks" / "img2" / "frame_0002.png");
    write_text_file(dataset_dir / "sparse" / "0" / "cameras.txt",
                    "1 PINHOLE 1 1 1 1 0.5 0.5\n");

    const lfs::io::DatasetInfo info = lfs::io::detect_dataset_info(dataset_dir);

    EXPECT_TRUE(fs::equivalent(info.images_path, dataset_dir / "images"));
    EXPECT_TRUE(fs::equivalent(info.masks_path, dataset_dir / "masks"));
    EXPECT_EQ(info.image_count, 2);
    EXPECT_TRUE(info.has_masks);
    EXPECT_EQ(info.mask_count, 2);
}
