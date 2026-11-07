// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/object_pool.h"

namespace {
namespace IR = Shader::IR;

constexpr u32 OpExtInstImport = 11;
constexpr u32 OpExtInst = 12;
constexpr u32 GLSLstd450FMin = 37;
constexpr u32 GLSLstd450FMax = 40;
constexpr u32 GLSLstd450FClamp = 43;
constexpr u32 GLSLstd450NMin = 79;
constexpr u32 GLSLstd450NMax = 80;
constexpr u32 GLSLstd450NClamp = 81;

struct GlslOpCounts {
    size_t fmin{};
    size_t fmax{};
    size_t fclamp{};
    size_t nmin{};
    size_t nmax{};
    size_t nclamp{};
};

GlslOpCounts CountGlslOps(const std::vector<u32>& code) {
    constexpr std::string_view glsl{"GLSL.std.450"};
    u32 glsl_set{};
    GlslOpCounts counts;
    for (size_t offset = 5; offset < code.size();) {
        const u32 word_count = code[offset] >> 16;
        REQUIRE(word_count > 0);
        REQUIRE(offset + word_count <= code.size());
        const u32 opcode = code[offset] & 0xffff;
        if (opcode == OpExtInstImport && word_count >= 5) {
            const auto* name = reinterpret_cast<const char*>(&code[offset + 2]);
            if (std::strncmp(name, glsl.data(), glsl.size()) == 0) {
                glsl_set = code[offset + 1];
            }
        } else if (opcode == OpExtInst && word_count >= 5 && code[offset + 3] == glsl_set) {
            switch (code[offset + 4]) {
            case GLSLstd450FMin:
                ++counts.fmin;
                break;
            case GLSLstd450FMax:
                ++counts.fmax;
                break;
            case GLSLstd450FClamp:
                ++counts.fclamp;
                break;
            case GLSLstd450NMin:
                ++counts.nmin;
                break;
            case GLSLstd450NMax:
                ++counts.nmax;
                break;
            case GLSLstd450NClamp:
                ++counts.nclamp;
                break;
            default:
                break;
            }
        }
        offset += word_count;
    }
    return counts;
}

struct Fixture {
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    IR::Program program;
    Shader::Profile profile{};

    Fixture() {
        program.stage = Shader::Stage::Compute;
        program.workgroup_size = {1, 1, 1};
        program.blocks = {&block};
        program.post_order_blocks = {&block};
        program.syntax_list = {{{.block = &block}, IR::AbstractSyntaxNode::Type::Block},
                               {{}, IR::AbstractSyntaxNode::Type::Return}};
        profile.supported_spirv = 0x00010300;
    }

    std::vector<u32> Emit(const char* name = "min-max") {
        auto code = Shader::Backend::SPIRV::EmitSPIRV(profile, program);
        if (const char* directory = std::getenv("EDEN_NAN_TEST_OUTPUT")) {
            std::filesystem::create_directories(directory);
            std::ofstream file{std::filesystem::path{directory} / (std::string{name} + ".spv"),
                               std::ios::binary};
            file.write(reinterpret_cast<const char*>(code.data()),
                       static_cast<std::streamsize>(code.size() * sizeof(u32)));
            REQUIRE(file.good());
        }
        return code;
    }
};

void CheckNaNDefined(const GlslOpCounts& counts, size_t nmin, size_t nmax) {
    CHECK(counts.fmin == 0);
    CHECK(counts.fmax == 0);
    CHECK(counts.fclamp == 0);
    CHECK(counts.nclamp == 0);
    CHECK(counts.nmin == nmin);
    CHECK(counts.nmax == nmax);
}
} // Anonymous namespace

TEST_CASE("SPIR-V float min, max and clamp emit NaN-defined instructions",
          "[video_core][spirv][nan_safe_min_max]") {
    for (const bool broken_clamp : {false, true}) {
        CAPTURE(broken_clamp);
        Fixture fixture;
        fixture.profile.has_broken_spirv_clamp = broken_clamp;
        fixture.program.info.uses_fp64 = true;
        fixture.program.info.uses_local_invocation_id = true;
        IR::IREmitter ir{fixture.block};
        const IR::F32 a{ir.BitCast<IR::F32>(ir.LocalInvocationIdX())};
        const IR::F32 b{ir.BitCast<IR::F32>(ir.LocalInvocationIdY())};
        const IR::F32 c{ir.BitCast<IR::F32>(ir.LocalInvocationIdZ())};
        const IR::F64 d{ir.PackDouble2x32(
            ir.CompositeConstruct(ir.LocalInvocationIdX(), ir.LocalInvocationIdY()))};
        const IR::F64 e{ir.PackDouble2x32(
            ir.CompositeConstruct(ir.LocalInvocationIdY(), ir.LocalInvocationIdZ()))};
        std::ignore = ir.FPMin(a, b);
        std::ignore = ir.FPMax(a, b);
        std::ignore = ir.FPMin(d, e);
        std::ignore = ir.FPMax(d, e);
        std::ignore = ir.FPSaturate(a);
        std::ignore = ir.FPClamp(a, b, c);
        // Min32 and min64, plus the min half of one saturate and one clamp; likewise for max.
        CheckNaNDefined(CountGlslOps(fixture.Emit()), 4, 4);
    }
}

TEST_CASE("SPIR-V float atomic min and max loops emit NaN-defined instructions",
          "[video_core][spirv][nan_safe_min_max]") {
    Fixture fixture;
    fixture.program.info.storage_buffers_descriptors = {{0, 0, 1, true}};
    fixture.program.info.uses_atomic_f32x2_min = true;
    fixture.program.info.uses_atomic_f32x2_max = true;
    CheckNaNDefined(CountGlslOps(fixture.Emit("atomic-min-max")), 1, 1);
}

TEST_CASE("SPIR-V saturation applies the lower bound before the upper bound",
          "[video_core][spirv][nan_safe_min_max]") {
    for (const bool broken_clamp : {false, true}) {
        Fixture fixture;
        fixture.profile.has_broken_spirv_clamp = broken_clamp;
        fixture.profile.unified_descriptor_binding = true;
        fixture.program.info.storage_buffers_descriptors = {{0, 0, 1, true}};
        fixture.program.info.used_storage_buffer_types = IR::Type::U32;
        IR::IREmitter ir{fixture.block};
        // One f32 input at byte 0; min(x,.5), max(x,.5), saturate(x), clamp(x,.25,.75)
        // at bytes 4..16. Keep this executable module for driver-side NaN/Inf checks.
        const IR::U32 bits{IR::Value{&*fixture.block.PrependNewInst(
            fixture.block.end(), IR::Opcode::LoadStorage32, {ir.Imm32(0), ir.Imm32(0)})}};
        const IR::F32 input{ir.BitCast<IR::F32>(bits)};
        const std::array<IR::F32, 4> results{
            IR::F32{ir.FPMin(input, ir.Imm32(0.5f))},
            IR::F32{ir.FPMax(input, ir.Imm32(0.5f))},
            IR::F32{ir.FPSaturate(input)},
            IR::F32{ir.FPClamp(input, ir.Imm32(0.25f), ir.Imm32(0.75f))}};
        for (u32 i = 0; i < results.size(); ++i) {
            fixture.block.AppendNewInst(IR::Opcode::WriteStorage32,
                {ir.Imm32(0), ir.Imm32(4 * (i + 1)), ir.BitCast<IR::U32>(results[i])});
        }
        const auto code = fixture.Emit(broken_clamp ? "nan-numeric-broken-clamp" : "nan-numeric");
        CheckNaNDefined(CountGlslOps(code), 3, 3);
        u32 previous_instruction{}, previous_result{};
        size_t chains{};
        for (size_t offset = 5; offset < code.size(); offset += code[offset] >> 16) {
            if ((code[offset] & 0xffff) != OpExtInst) {
                continue;
            }
            const u32 instruction = code[offset + 4];
            if (instruction == GLSLstd450NMin && previous_instruction == GLSLstd450NMax) {
                CHECK(code[offset + 5] == previous_result);
                ++chains;
            }
            previous_instruction = instruction;
            previous_result = code[offset + 2];
        }
        CHECK(chains == 2);
    }
}
