#include <bx/allocator.h>
#include <bx/debug.h>
#include <bx/math.h>
#include <bx/file.h>
#include <bimg/decode.h>
#include <bimg/encode.h>

#include <iostream>

#include "common.h"
#include "bgfx_utils.h"
#include "imgui/imgui.h"
#include "camera.h"

namespace
{

	static const uint16_t s_terrainSize = 256;

	struct PosTexCoord0Vertex
	{
		float m_x;
		float m_y;
		float m_z;
		float m_u;
		float m_v;

		static void init()
		{
			ms_layout
				.begin()
				.add(bgfx::Attrib::Position, 3, bgfx::AttribType::Float)
				.add(bgfx::Attrib::TexCoord0, 2, bgfx::AttribType::Float)
				.end();
		}

		static bgfx::VertexLayout ms_layout;
	};

	bgfx::VertexLayout PosTexCoord0Vertex::ms_layout;

	struct TerrainData
	{
		int32_t              m_mode;
		bool                 m_dirty;
		float                m_transform[16];
		uint8_t* m_heightMap;

		PosTexCoord0Vertex* m_vertices;
		uint32_t             m_vertexCount;
		uint16_t* m_indices;
		uint32_t             m_indexCount;
	};

	struct BrushData
	{
		bool    m_raise;
		int32_t m_size;
		float   m_power;
	};

	// New structure for vegetation parameters
	struct VegetationData
	{
		bool  m_enable;
		float m_scale;
		float m_grassThreshold;
		float m_rockThreshold;
		float m_snowThreshold;
		float m_blendSharpness;
	};

	class ExampleTerrain : public entry::AppI
	{
	public:
		ExampleTerrain(const char* _name, const char* _description, const char* _url)
			: entry::AppI(_name, _description, _url)
		{
		}
		// 从标准图像格式(PNG、JPG等)加载纹理
		bgfx::TextureHandle loadTextureFromFile(const char* _filePath, uint64_t _flags = BGFX_TEXTURE_NONE | BGFX_SAMPLER_NONE)
		{
			bgfx::TextureHandle handle = BGFX_INVALID_HANDLE;

			// 打开文件
			bx::FileReader reader;
			if (!bx::open(&reader, _filePath))
			{
				printf("Failed to open image file: %s\n", _filePath);
				return handle;
			}

			// 获取文件大小并读取数据
			uint32_t size = (uint32_t)bx::getSize(&reader);
			void* data = bx::alloc(entry::getAllocator(), size);

			bx::Error err;
			bx::read(&reader, data, size, &err);
			bx::close(&reader);

			if (err.isOk())
			{
				// 解码图像
				bimg::ImageContainer* imageContainer = bimg::imageParse(entry::getAllocator(), data, size);
				bx::free(entry::getAllocator(), data);

				if (nullptr != imageContainer)
				{
					// 将图像转换为RGBA8格式，这是最通用的格式
					bimg::ImageContainer* rgba8Container = bimg::imageConvert(entry::getAllocator(),
						bimg::TextureFormat::RGBA8,
						*imageContainer);

					// 如果转换成功，释放原始容器，使用转换后的容器
					if (rgba8Container != nullptr)
					{
						bimg::imageFree(imageContainer);
						imageContainer = rgba8Container;
					}

					// 创建内存引用
					const bgfx::Memory* mem = bgfx::copy(imageContainer->m_data, imageContainer->m_size);

					// 创建纹理
					handle = bgfx::createTexture2D(
						uint16_t(imageContainer->m_width),
						uint16_t(imageContainer->m_height),
						false, // 不使用mip maps简化处理
						1,     // 只用一层
						bgfx::TextureFormat::RGBA8, // 使用RGBA8格式
						_flags,
						mem
					);

					// 释放图像容器
					bimg::imageFree(imageContainer);

					if (!bgfx::isValid(handle))
					{
						printf("Failed to create texture from: %s\n", _filePath);
					}
				}
				else
				{
					printf("Failed to parse image: %s\n", _filePath);
				}
			}
			else
			{
				printf("Failed to read image file: %s\n", _filePath);
				bx::free(entry::getAllocator(), data);
			}

			return handle;
		}

		void init(int32_t _argc, const char* const* _argv, uint32_t _width, uint32_t _height) override
		{
			Args args(_argc, _argv);

			m_width = _width;
			m_height = _height;
			m_debug = BGFX_DEBUG_NONE;
			m_reset = BGFX_RESET_VSYNC;

			bgfx::Init init;
			init.type = args.m_type;
			init.vendorId = args.m_pciId;
			init.platformData.nwh = entry::getNativeWindowHandle(entry::kDefaultWindowHandle);
			init.platformData.ndt = entry::getNativeDisplayHandle();
			init.platformData.type = entry::getNativeWindowHandleType();
			init.resolution.width = m_width;
			init.resolution.height = m_height;
			init.resolution.reset = m_reset;
			bgfx::init(init);

			// Enable m_debug text.
			bgfx::setDebug(m_debug);

			// Set view 0 clear state.
			bgfx::setViewClear(0
				, BGFX_CLEAR_COLOR | BGFX_CLEAR_DEPTH
				, 0x303030ff
				, 1.0f
				, 0
			);

			// Create vertex stream declaration.
			PosTexCoord0Vertex::init();

			// Create program from shaders.
			m_terrainProgram = loadProgram("vs_terrain", "fs_terrain");
			m_terrainHeightTextureProgram = loadProgram("vs_terrain_height_texture", "fs_terrain");

			// Load the enhanced shaders with vegetation support
			m_terrainVegetationProgram = loadProgram("vs_terrain", "fs_terrain_vegetation");
			m_terrainHeightTextureVegetationProgram = loadProgram("vs_terrain_height_texture", "fs_terrain_vegetation");

			// Imgui.
			imguiCreate();

			m_timeOffset = bx::getHPCounter();

			m_vbh.idx = bgfx::kInvalidHandle;
			m_ibh.idx = bgfx::kInvalidHandle;
			m_dvbh.idx = bgfx::kInvalidHandle;
			m_dibh.idx = bgfx::kInvalidHandle;
			m_heightTexture.idx = bgfx::kInvalidHandle;
			s_heightTexture = bgfx::createUniform("s_heightTexture", bgfx::UniformType::Sampler);

			// Create uniform handles for vegetation textures
			s_grassTexture = bgfx::createUniform("s_grassTexture", bgfx::UniformType::Sampler);
			s_rockTexture = bgfx::createUniform("s_rockTexture", bgfx::UniformType::Sampler);
			s_snowTexture = bgfx::createUniform("s_snowTexture", bgfx::UniformType::Sampler);
			u_vegetationParams = bgfx::createUniform("u_vegetationParams", bgfx::UniformType::Vec4);

			// Load vegetation textures
			m_grassTexture = loadTexture("textures/grass.dds");
			m_rockTexture = loadTexture("textures/rock.dds");
			m_snowTexture = loadTexture("textures/snow.dds");

			m_oldWidth = 0;
			m_oldHeight = 0;
			m_oldReset = m_reset;

			m_brush.m_power = 0.5f;
			m_brush.m_size = 10;
			m_brush.m_raise = true;

			// Initialize vegetation parameters
			m_vegetation.m_enable = true;
			m_vegetation.m_scale = 32.0f;
			m_vegetation.m_grassThreshold = 0.3f;
			m_vegetation.m_rockThreshold = 0.6f;
			m_vegetation.m_snowThreshold = 0.8f;
			m_vegetation.m_blendSharpness = 0.2f;

			uint32_t num = s_terrainSize * s_terrainSize;

			m_terrain.m_mode = 0;
			m_terrain.m_dirty = true;
			m_terrain.m_vertices = (PosTexCoord0Vertex*)bx::alloc(entry::getAllocator(), num * sizeof(PosTexCoord0Vertex));
			m_terrain.m_indices = (uint16_t*)bx::alloc(entry::getAllocator(), num * sizeof(uint16_t) * 6);
			m_terrain.m_heightMap = (uint8_t*)bx::alloc(entry::getAllocator(), num);

			bx::mtxSRT(m_terrain.m_transform, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
			bx::memSet(m_terrain.m_heightMap, 0, sizeof(uint8_t) * s_terrainSize * s_terrainSize);

			cameraCreate();

			cameraSetPosition({ s_terrainSize / 2.0f, 100.0f, 0.0f });
			cameraSetVerticalAngle(-bx::kPiQuarter);
		}

		virtual int shutdown() override
		{
			// Cleanup.
			cameraDestroy();
			imguiDestroy();

			if (bgfx::isValid(m_ibh))
			{
				bgfx::destroy(m_ibh);
			}

			if (bgfx::isValid(m_vbh))
			{
				bgfx::destroy(m_vbh);
			}

			if (bgfx::isValid(m_dibh))
			{
				bgfx::destroy(m_dibh);
			}

			if (bgfx::isValid(m_dvbh))
			{
				bgfx::destroy(m_dvbh);
			}

			bgfx::destroy(s_heightTexture);

			// Destroy vegetation related resources
			bgfx::destroy(s_grassTexture);
			bgfx::destroy(s_rockTexture);
			bgfx::destroy(s_snowTexture);
			bgfx::destroy(u_vegetationParams);

			if (bgfx::isValid(m_grassTexture))
			{
				bgfx::destroy(m_grassTexture);
			}

			if (bgfx::isValid(m_rockTexture))
			{
				bgfx::destroy(m_rockTexture);
			}

			if (bgfx::isValid(m_snowTexture))
			{
				bgfx::destroy(m_snowTexture);
			}

			if (bgfx::isValid(m_heightTexture))
			{
				bgfx::destroy(m_heightTexture);
			}

			bgfx::destroy(m_terrainProgram);
			bgfx::destroy(m_terrainHeightTextureProgram);
			bgfx::destroy(m_terrainVegetationProgram);
			bgfx::destroy(m_terrainHeightTextureVegetationProgram);

			/// When data is passed to bgfx via makeRef we need to make
			/// sure library is done with it before freeing memory blocks.
			bgfx::frame();

			bx::AllocatorI* allocator = entry::getAllocator();
			bx::free(allocator, m_terrain.m_vertices);
			bx::free(allocator, m_terrain.m_indices);
			bx::free(allocator, m_terrain.m_heightMap);

			// Shutdown bgfx.
			bgfx::shutdown();

			return 0;
		}

		void updateTerrainMesh()
		{
			m_terrain.m_vertexCount = 0;
			for (uint32_t y = 0; y < s_terrainSize; y++)
			{
				for (uint32_t x = 0; x < s_terrainSize; x++)
				{
					PosTexCoord0Vertex* vert = &m_terrain.m_vertices[m_terrain.m_vertexCount];
					vert->m_x = (float)x;
					vert->m_y = m_terrain.m_heightMap[(y * s_terrainSize) + x];
					vert->m_z = (float)y;

					// Set texture coordinates based on vegetation scale
					vert->m_u = (x) / m_vegetation.m_scale;
					vert->m_v = (y) / m_vegetation.m_scale;

					m_terrain.m_vertexCount++;
				}
			}

			m_terrain.m_indexCount = 0;
			for (uint16_t y = 0; y < (s_terrainSize - 1); y++)
			{
				uint16_t y_offset = (y * s_terrainSize);
				for (uint16_t x = 0; x < (s_terrainSize - 1); x++)
				{
					m_terrain.m_indices[m_terrain.m_indexCount + 0] = y_offset + x + 1;
					m_terrain.m_indices[m_terrain.m_indexCount + 1] = y_offset + x + s_terrainSize;
					m_terrain.m_indices[m_terrain.m_indexCount + 2] = y_offset + x;
					m_terrain.m_indices[m_terrain.m_indexCount + 3] = y_offset + x + s_terrainSize + 1;
					m_terrain.m_indices[m_terrain.m_indexCount + 4] = y_offset + x + s_terrainSize;
					m_terrain.m_indices[m_terrain.m_indexCount + 5] = y_offset + x + 1;

					m_terrain.m_indexCount += 6;
				}
			}
		}

		void updateTerrain()
		{
			const bgfx::Memory* mem;

			switch (m_terrain.m_mode)
			{
			default: // Vertex Buffer : Destroy and recreate a regular vertex buffer to update terrain.
				updateTerrainMesh();

				if (bgfx::isValid(m_vbh))
				{
					bgfx::destroy(m_vbh);
				}

				mem = bgfx::makeRef(&m_terrain.m_vertices[0], sizeof(PosTexCoord0Vertex) * m_terrain.m_vertexCount);
				m_vbh = bgfx::createVertexBuffer(mem, PosTexCoord0Vertex::ms_layout);
				if (bgfx::isValid(m_ibh))
				{
					bgfx::destroy(m_ibh);
				}

				mem = bgfx::makeRef(&m_terrain.m_indices[0], sizeof(uint16_t) * m_terrain.m_indexCount);
				m_ibh = bgfx::createIndexBuffer(mem);
				break;

			case 1: // Dynamic Vertex Buffer : Utilize dynamic vertex buffer to update terrain.
				updateTerrainMesh();

				if (!bgfx::isValid(m_dvbh))
				{
					m_dvbh = bgfx::createDynamicVertexBuffer(m_terrain.m_vertexCount, PosTexCoord0Vertex::ms_layout);
				}

				mem = bgfx::makeRef(&m_terrain.m_vertices[0], sizeof(PosTexCoord0Vertex) * m_terrain.m_vertexCount);
				bgfx::update(m_dvbh, 0, mem);

				if (!bgfx::isValid(m_dibh))
				{
					m_dibh = bgfx::createDynamicIndexBuffer(m_terrain.m_indexCount);
				}

				mem = bgfx::makeRef(&m_terrain.m_indices[0], sizeof(uint16_t) * m_terrain.m_indexCount);
				bgfx::update(m_dibh, 0, mem);
				break;

			case 2: // Height Texture: Update a height texture that is sampled in the terrain vertex shader.
				if (!bgfx::isValid(m_vbh) || !bgfx::isValid(m_ibh))
				{
					updateTerrainMesh();

					mem = bgfx::makeRef(&m_terrain.m_vertices[0], sizeof(PosTexCoord0Vertex) * m_terrain.m_vertexCount);
					m_vbh = bgfx::createVertexBuffer(mem, PosTexCoord0Vertex::ms_layout);

					mem = bgfx::makeRef(&m_terrain.m_indices[0], sizeof(uint16_t) * m_terrain.m_indexCount);
					m_ibh = bgfx::createIndexBuffer(mem);
				}

				if (!bgfx::isValid(m_heightTexture))
				{
					// m_heightTexture = bgfx::createTexture2D(s_terrainSize, s_terrainSize, false, bgfx::TextureFormat::D32, bgfx::TextureFormat::R8);
					m_heightTexture = loadTextureFromFile("textures/0010.png", BGFX_SAMPLER_MIN_ANISOTROPIC | BGFX_SAMPLER_MAG_ANISOTROPIC);
				}

				mem = bgfx::makeRef(&m_terrain.m_heightMap[0], sizeof(uint8_t) * s_terrainSize * s_terrainSize);
				bgfx::updateTexture2D(m_heightTexture, 0, 0, 0, 0, s_terrainSize, s_terrainSize, mem);
				break;
			}
		}

		void paintTerrainHeight(uint32_t _x, uint32_t _y)
		{
			for (int32_t area_y = -m_brush.m_size; area_y < m_brush.m_size; ++area_y)
			{
				for (int32_t area_x = -m_brush.m_size; area_x < m_brush.m_size; ++area_x)
				{
					int32_t brush_x = _x + area_x;
					if (brush_x < 0
						|| brush_x >= (int32_t)s_terrainSize)
					{
						continue;
					}

					int32_t brush_y = _y + area_y;
					if (brush_y < 0
						|| brush_y >= (int32_t)s_terrainSize)
					{
						continue;
					}

					uint32_t heightMapPos = (brush_y * s_terrainSize) + brush_x;
					float height = (float)m_terrain.m_heightMap[heightMapPos];

					// Brush attenuation
					float a2 = (float)(area_x * area_x);
					float b2 = (float)(area_y * area_y);
					float brushAttn = m_brush.m_size - bx::sqrt(a2 + b2);

					// Raise/Lower and scale by brush power.
					height += 0.0f < bx::clamp(brushAttn * m_brush.m_power, 0.0f, m_brush.m_power) && m_brush.m_raise
						? 1.0f
						: -1.0f
						;

					m_terrain.m_heightMap[heightMapPos] = (uint8_t)bx::clamp(height, 0.0f, 255.0f);
					m_terrain.m_dirty = true;
				}
			}
		}

		void mousePickTerrain()
		{
			float ray_clip[4];
			ray_clip[0] = ((2.0f * m_mouseState.m_mx) / m_width - 1.0f) * -1.0f;
			ray_clip[1] = ((1.0f - (2.0f * m_mouseState.m_my) / m_height)) * -1.0f;
			ray_clip[2] = -1.0f;
			ray_clip[3] = 1.0f;

			float invProjMtx[16];
			bx::mtxInverse(invProjMtx, m_projMtx);

			float ray_eye[4];
			bx::vec4MulMtx(ray_eye, ray_clip, invProjMtx);
			ray_eye[2] = -1.0f;
			ray_eye[3] = 0.0f;

			float invViewMtx[16];
			bx::mtxInverse(invViewMtx, m_viewMtx);

			float ray_world[4];
			bx::vec4MulMtx(ray_world, ray_eye, invViewMtx);

			const bx::Vec3 rayDir = bx::mul(bx::normalize(bx::load<bx::Vec3>(ray_world)), -1.0f);

			bx::Vec3 pos = cameraGetPosition();
			for (int i = 0; i < 1000; ++i)
			{
				pos = bx::add(pos, rayDir);

				if (pos.x < 0
					|| pos.x >= s_terrainSize
					|| pos.z < 0
					|| pos.z >= s_terrainSize)
				{
					continue;
				}

				uint32_t heightMapPos = ((uint32_t)pos.z * s_terrainSize) + (uint32_t)pos.x;
				if (pos.y < m_terrain.m_heightMap[heightMapPos])
				{
					paintTerrainHeight((uint32_t)pos.x, (uint32_t)pos.z);
					return;
				}
			}
		}

		bool update() override
		{
			if (!entry::processEvents(m_width, m_height, m_debug, m_reset, &m_mouseState))
			{
				int64_t now = bx::getHPCounter();
				static int64_t last = now;
				const int64_t frameTime = now - last;
				last = now;
				const double freq = double(bx::getHPFrequency());
				const float deltaTime = float(frameTime / freq);

				imguiBeginFrame(m_mouseState.m_mx
					, m_mouseState.m_my
					, (m_mouseState.m_buttons[entry::MouseButton::Left] ? IMGUI_MBUT_LEFT : 0)
					| (m_mouseState.m_buttons[entry::MouseButton::Right] ? IMGUI_MBUT_RIGHT : 0)
					| (m_mouseState.m_buttons[entry::MouseButton::Middle] ? IMGUI_MBUT_MIDDLE : 0)
					, m_mouseState.m_mz
					, uint16_t(m_width)
					, uint16_t(m_height)
				);

				showExampleDialog(this);

				ImGui::SetNextWindowPos(
					ImVec2(m_width - m_width / 5.0f - 10.0f, 10.0f)
					, ImGuiCond_FirstUseEver
				);
				ImGui::SetNextWindowSize(
					ImVec2(m_width / 5.0f, m_height / 3.0f)
					, ImGuiCond_FirstUseEver
				);
				ImGui::Begin("Settings"
					, NULL
					, 0
				);

				ImGui::Separator();

				m_terrain.m_dirty |= ImGui::RadioButton("Vertex Buffer", &m_terrain.m_mode, 0);
				m_terrain.m_dirty |= ImGui::RadioButton("Dynamic Vertex Buffer", &m_terrain.m_mode, 1);
				m_terrain.m_dirty |= ImGui::RadioButton("Height Texture", &m_terrain.m_mode, 2);

				ImGui::Separator();

				ImGui::Checkbox("Raise Terrain", &m_brush.m_raise);

				ImGui::SliderInt("Brush Size", &m_brush.m_size, 1, 50);
				ImGui::SliderFloat("Brush Power", &m_brush.m_power, 0.0f, 1.0f);

				ImGui::Separator();

				// Vegetation settings
				if (ImGui::CollapsingHeader("Vegetation Settings", ImGuiTreeNodeFlags_DefaultOpen))
				{
					m_terrain.m_dirty |= ImGui::Checkbox("Enable Vegetation", &m_vegetation.m_enable);
					m_terrain.m_dirty |= ImGui::SliderFloat("Texture Scale", &m_vegetation.m_scale, 8.0f, 128.0f);
					m_terrain.m_dirty |= ImGui::SliderFloat("Grass Threshold", &m_vegetation.m_grassThreshold, 0.0f, 1.0f);
					m_terrain.m_dirty |= ImGui::SliderFloat("Rock Threshold", &m_vegetation.m_rockThreshold, 0.0f, 1.0f);
					m_terrain.m_dirty |= ImGui::SliderFloat("Snow Threshold", &m_vegetation.m_snowThreshold, 0.0f, 1.0f);
					m_terrain.m_dirty |= ImGui::SliderFloat("Blend Sharpness", &m_vegetation.m_blendSharpness, 0.01f, 1.0f);
				}

				ImGui::End();
				imguiEndFrame();

				// Update camera.
				cameraUpdate(deltaTime, m_mouseState, ImGui::MouseOverArea());

				if (!ImGui::MouseOverArea())
				{
					if (!!m_mouseState.m_buttons[entry::MouseButton::Left])
					{
						mousePickTerrain();
					}
				}

				// Update terrain.
				if (m_terrain.m_dirty)
				{
					updateTerrain();
					m_terrain.m_dirty = false;
				}

				// Set view 0 default viewport.
				bgfx::setViewRect(0, 0, 0, uint16_t(m_width), uint16_t(m_height));

				cameraGetViewMtx(m_viewMtx);
				bx::mtxProj(m_projMtx, 60.0f, float(m_width) / float(m_height), 0.1f, 2000.0f, bgfx::getCaps()->homogeneousDepth);

				bgfx::setViewTransform(0, m_viewMtx, m_projMtx);
				bgfx::setTransform(m_terrain.m_transform);

				// Set vegetation parameters for the shader
				float vegetationParams[4] = {
					m_vegetation.m_grassThreshold,
					m_vegetation.m_rockThreshold,
					m_vegetation.m_snowThreshold,
					m_vegetation.m_blendSharpness
				};
				bgfx::setUniform(u_vegetationParams, vegetationParams);

				switch (m_terrain.m_mode)
				{
				default:
					bgfx::setVertexBuffer(0, m_vbh);
					bgfx::setIndexBuffer(m_ibh);
					if (m_vegetation.m_enable) {
						// Set vegetation textures
						bgfx::setTexture(0, s_grassTexture, m_grassTexture);
						bgfx::setTexture(1, s_rockTexture, m_rockTexture);
						bgfx::setTexture(2, s_snowTexture, m_snowTexture);
						bgfx::submit(0, m_terrainVegetationProgram);
					}
					else {
						bgfx::submit(0, m_terrainProgram);
					}
					break;

				case 1:
					bgfx::setVertexBuffer(0, m_dvbh);
					bgfx::setIndexBuffer(m_dibh);
					if (m_vegetation.m_enable) {
						// Set vegetation textures
						bgfx::setTexture(0, s_grassTexture, m_grassTexture);
						bgfx::setTexture(1, s_rockTexture, m_rockTexture);
						bgfx::setTexture(2, s_snowTexture, m_snowTexture);
						bgfx::submit(0, m_terrainVegetationProgram);
					}
					else {
						bgfx::submit(0, m_terrainProgram);
					}
					break;

				case 2:
					bgfx::setVertexBuffer(0, m_vbh);
					bgfx::setIndexBuffer(m_ibh);
					bgfx::setTexture(3, s_heightTexture, m_heightTexture);
					if (m_vegetation.m_enable) {
						// Set vegetation textures
						bgfx::setTexture(0, s_grassTexture, m_grassTexture);
						bgfx::setTexture(1, s_rockTexture, m_rockTexture);
						bgfx::setTexture(2, s_snowTexture, m_snowTexture);
						bgfx::submit(0, m_terrainHeightTextureVegetationProgram);
					}
					else {
						bgfx::submit(0, m_terrainHeightTextureProgram);
					}
					break;
				}

				// Advance to next frame. Rendering thread will be kicked to
				// process submitted rendering primitives.
				bgfx::frame();

				return true;
			}

			return false;
		}

		bgfx::VertexBufferHandle m_vbh;
		bgfx::IndexBufferHandle m_ibh;
		bgfx::DynamicVertexBufferHandle m_dvbh;
		bgfx::DynamicIndexBufferHandle m_dibh;
		bgfx::ProgramHandle m_terrainProgram;
		bgfx::ProgramHandle m_terrainHeightTextureProgram;
		bgfx::ProgramHandle m_terrainVegetationProgram;
		bgfx::ProgramHandle m_terrainHeightTextureVegetationProgram;
		bgfx::UniformHandle s_heightTexture;
		bgfx::TextureHandle m_heightTexture;

		// Vegetation related handles
		bgfx::UniformHandle s_grassTexture;
		bgfx::UniformHandle s_rockTexture;
		bgfx::UniformHandle s_snowTexture;
		bgfx::UniformHandle u_vegetationParams;
		bgfx::TextureHandle m_grassTexture;
		bgfx::TextureHandle m_rockTexture;
		bgfx::TextureHandle m_snowTexture;

		float m_viewMtx[16];
		float m_projMtx[16];

		uint32_t m_width;
		uint32_t m_height;
		uint32_t m_debug;
		uint32_t m_reset;

		uint32_t m_oldWidth;
		uint32_t m_oldHeight;
		uint32_t m_oldReset;

		TerrainData m_terrain;
		BrushData m_brush;
		VegetationData m_vegetation;

		entry::MouseState m_mouseState;

		int64_t m_timeOffset;
	};

} // namespace

ENTRY_IMPLEMENT_MAIN(
	ExampleTerrain
	, "27-terrain"
	, "Terrain painting example."
	, "https://bkaradzic.github.io/bgfx/examples.html#terrain"
);
