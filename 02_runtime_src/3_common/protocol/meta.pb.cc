// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: meta.proto

#include "meta.pb.h"

#include <algorithm>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
extern PROTOBUF_INTERNAL_EXPORT_common_2eproto ::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<0> scc_info_Image_common_2eproto;
extern PROTOBUF_INTERNAL_EXPORT_meta_5fdata_2eproto ::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<5> scc_info_Data_meta_5fdata_2eproto;
namespace Meta {
class MetaDefaultTypeInternal {
 public:
  ::PROTOBUF_NAMESPACE_ID::internal::ExplicitlyConstructed<Meta> _instance;
} _Meta_default_instance_;
}  // namespace Meta
static void InitDefaultsscc_info_Meta_meta_2eproto() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::Meta::_Meta_default_instance_;
    new (ptr) ::Meta::Meta();
    ::PROTOBUF_NAMESPACE_ID::internal::OnShutdownDestroyMessage(ptr);
  }
  ::Meta::Meta::InitAsDefaultInstance();
}

::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<2> scc_info_Meta_meta_2eproto =
    {{ATOMIC_VAR_INIT(::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase::kUninitialized), 2, InitDefaultsscc_info_Meta_meta_2eproto}, {
      &scc_info_Image_common_2eproto.base,
      &scc_info_Data_meta_5fdata_2eproto.base,}};

static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_meta_2eproto[1];
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const** file_level_enum_descriptors_meta_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_meta_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_meta_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  PROTOBUF_FIELD_OFFSET(::Meta::Meta, _has_bits_),
  PROTOBUF_FIELD_OFFSET(::Meta::Meta, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::Meta::Meta, version_),
  PROTOBUF_FIELD_OFFSET(::Meta::Meta, frame_id_),
  PROTOBUF_FIELD_OFFSET(::Meta::Meta, img_frame_),
  PROTOBUF_FIELD_OFFSET(::Meta::Meta, proto_version_),
  PROTOBUF_FIELD_OFFSET(::Meta::Meta, data_),
  2,
  3,
  0,
  4,
  1,
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, 10, sizeof(::Meta::Meta)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::Meta::_Meta_default_instance_),
};

const char descriptor_table_protodef_meta_2eproto[] =
  "\n\nmeta.proto\022\004Meta\032\017meta_data.proto\032\014com"
  "mon.proto\"\210\001\n\004Meta\022\017\n\007version\030\001 \002(\005\022\020\n\010f"
  "rame_id\030\002 \002(\005\022%\n\timg_frame\030\004 \001(\0132\022.Commo"
  "nProto.Image\022\030\n\rproto_version\030\005 \001(\005:\0011\022\034"
  "\n\004data\030\007 \001(\0132\016.MetaData.Data"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_meta_2eproto_deps[2] = {
  &::descriptor_table_common_2eproto,
  &::descriptor_table_meta_5fdata_2eproto,
};
static ::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase*const descriptor_table_meta_2eproto_sccs[1] = {
  &scc_info_Meta_meta_2eproto.base,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_meta_2eproto_once;
static bool descriptor_table_meta_2eproto_initialized = false;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_meta_2eproto = {
  &descriptor_table_meta_2eproto_initialized, descriptor_table_protodef_meta_2eproto, "meta.proto", 188,
  &descriptor_table_meta_2eproto_once, descriptor_table_meta_2eproto_sccs, descriptor_table_meta_2eproto_deps, 1, 2,
  schemas, file_default_instances, TableStruct_meta_2eproto::offsets,
  file_level_metadata_meta_2eproto, 1, file_level_enum_descriptors_meta_2eproto, file_level_service_descriptors_meta_2eproto,
};

// Force running AddDescriptors() at dynamic initialization time.
static bool dynamic_init_dummy_meta_2eproto = (  ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptors(&descriptor_table_meta_2eproto), true);
namespace Meta {

// ===================================================================

void Meta::InitAsDefaultInstance() {
  ::Meta::_Meta_default_instance_._instance.get_mutable()->img_frame_ = const_cast< ::CommonProto::Image*>(
      ::CommonProto::Image::internal_default_instance());
  ::Meta::_Meta_default_instance_._instance.get_mutable()->data_ = const_cast< ::MetaData::Data*>(
      ::MetaData::Data::internal_default_instance());
}
class Meta::HasBitSetters {
 public:
  using HasBits = decltype(std::declval<Meta>()._has_bits_);
  static void set_has_version(HasBits* has_bits) {
    (*has_bits)[0] |= 4u;
  }
  static void set_has_frame_id(HasBits* has_bits) {
    (*has_bits)[0] |= 8u;
  }
  static const ::CommonProto::Image& img_frame(const Meta* msg);
  static void set_has_img_frame(HasBits* has_bits) {
    (*has_bits)[0] |= 1u;
  }
  static void set_has_proto_version(HasBits* has_bits) {
    (*has_bits)[0] |= 16u;
  }
  static const ::MetaData::Data& data(const Meta* msg);
  static void set_has_data(HasBits* has_bits) {
    (*has_bits)[0] |= 2u;
  }
};

const ::CommonProto::Image&
Meta::HasBitSetters::img_frame(const Meta* msg) {
  return *msg->img_frame_;
}
const ::MetaData::Data&
Meta::HasBitSetters::data(const Meta* msg) {
  return *msg->data_;
}
void Meta::clear_img_frame() {
  if (img_frame_ != nullptr) img_frame_->Clear();
  _has_bits_[0] &= ~0x00000001u;
}
void Meta::clear_data() {
  if (data_ != nullptr) data_->Clear();
  _has_bits_[0] &= ~0x00000002u;
}
#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int Meta::kVersionFieldNumber;
const int Meta::kFrameIdFieldNumber;
const int Meta::kImgFrameFieldNumber;
const int Meta::kProtoVersionFieldNumber;
const int Meta::kDataFieldNumber;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

Meta::Meta()
  : ::PROTOBUF_NAMESPACE_ID::Message(), _internal_metadata_(nullptr) {
  SharedCtor();
  // @@protoc_insertion_point(constructor:Meta.Meta)
}
Meta::Meta(const Meta& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _internal_metadata_(nullptr),
      _has_bits_(from._has_bits_) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  if (from.has_img_frame()) {
    img_frame_ = new ::CommonProto::Image(*from.img_frame_);
  } else {
    img_frame_ = nullptr;
  }
  if (from.has_data()) {
    data_ = new ::MetaData::Data(*from.data_);
  } else {
    data_ = nullptr;
  }
  ::memcpy(&version_, &from.version_,
    static_cast<size_t>(reinterpret_cast<char*>(&proto_version_) -
    reinterpret_cast<char*>(&version_)) + sizeof(proto_version_));
  // @@protoc_insertion_point(copy_constructor:Meta.Meta)
}

void Meta::SharedCtor() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&scc_info_Meta_meta_2eproto.base);
  ::memset(&img_frame_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&frame_id_) -
      reinterpret_cast<char*>(&img_frame_)) + sizeof(frame_id_));
  proto_version_ = 1;
}

Meta::~Meta() {
  // @@protoc_insertion_point(destructor:Meta.Meta)
  SharedDtor();
}

void Meta::SharedDtor() {
  if (this != internal_default_instance()) delete img_frame_;
  if (this != internal_default_instance()) delete data_;
}

void Meta::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const Meta& Meta::default_instance() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&::scc_info_Meta_meta_2eproto.base);
  return *internal_default_instance();
}


void Meta::Clear() {
// @@protoc_insertion_point(message_clear_start:Meta.Meta)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000003u) {
    if (cached_has_bits & 0x00000001u) {
      GOOGLE_DCHECK(img_frame_ != nullptr);
      img_frame_->Clear();
    }
    if (cached_has_bits & 0x00000002u) {
      GOOGLE_DCHECK(data_ != nullptr);
      data_->Clear();
    }
  }
  if (cached_has_bits & 0x0000001cu) {
    ::memset(&version_, 0, static_cast<size_t>(
        reinterpret_cast<char*>(&frame_id_) -
        reinterpret_cast<char*>(&version_)) + sizeof(frame_id_));
    proto_version_ = 1;
  }
  _has_bits_.Clear();
  _internal_metadata_.Clear();
}

#if GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
const char* Meta::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  HasBitSetters::HasBits has_bits{};
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    CHK_(ptr);
    switch (tag >> 3) {
      // required int32 version = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 8)) {
          HasBitSetters::set_has_version(&has_bits);
          version_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // required int32 frame_id = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 16)) {
          HasBitSetters::set_has_frame_id(&has_bits);
          frame_id_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional .CommonProto.Image img_frame = 4;
      case 4:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 34)) {
          ptr = ctx->ParseMessage(mutable_img_frame(), ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional int32 proto_version = 5 [default = 1];
      case 5:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 40)) {
          HasBitSetters::set_has_proto_version(&has_bits);
          proto_version_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional .MetaData.Data data = 7;
      case 7:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 58)) {
          ptr = ctx->ParseMessage(mutable_data(), ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      default: {
      handle_unusual:
        if ((tag & 7) == 4 || tag == 0) {
          ctx->SetLastTag(tag);
          goto success;
        }
        ptr = UnknownFieldParse(tag, &_internal_metadata_, ptr, ctx);
        CHK_(ptr != nullptr);
        continue;
      }
    }  // switch
  }  // while
success:
  _has_bits_.Or(has_bits);
  return ptr;
failure:
  ptr = nullptr;
  goto success;
#undef CHK_
}
#else  // GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
bool Meta::MergePartialFromCodedStream(
    ::PROTOBUF_NAMESPACE_ID::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!PROTOBUF_PREDICT_TRUE(EXPRESSION)) goto failure
  ::PROTOBUF_NAMESPACE_ID::uint32 tag;
  // @@protoc_insertion_point(parse_start:Meta.Meta)
  for (;;) {
    ::std::pair<::PROTOBUF_NAMESPACE_ID::uint32, bool> p = input->ReadTagWithCutoffNoLastTag(127u);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // required int32 version = 1;
      case 1: {
        if (static_cast< ::PROTOBUF_NAMESPACE_ID::uint8>(tag) == (8 & 0xFF)) {
          HasBitSetters::set_has_version(&_has_bits_);
          DO_((::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::ReadPrimitive<
                   ::PROTOBUF_NAMESPACE_ID::int32, ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_INT32>(
                 input, &version_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // required int32 frame_id = 2;
      case 2: {
        if (static_cast< ::PROTOBUF_NAMESPACE_ID::uint8>(tag) == (16 & 0xFF)) {
          HasBitSetters::set_has_frame_id(&_has_bits_);
          DO_((::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::ReadPrimitive<
                   ::PROTOBUF_NAMESPACE_ID::int32, ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_INT32>(
                 input, &frame_id_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // optional .CommonProto.Image img_frame = 4;
      case 4: {
        if (static_cast< ::PROTOBUF_NAMESPACE_ID::uint8>(tag) == (34 & 0xFF)) {
          DO_(::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::ReadMessage(
               input, mutable_img_frame()));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // optional int32 proto_version = 5 [default = 1];
      case 5: {
        if (static_cast< ::PROTOBUF_NAMESPACE_ID::uint8>(tag) == (40 & 0xFF)) {
          HasBitSetters::set_has_proto_version(&_has_bits_);
          DO_((::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::ReadPrimitive<
                   ::PROTOBUF_NAMESPACE_ID::int32, ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_INT32>(
                 input, &proto_version_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // optional .MetaData.Data data = 7;
      case 7: {
        if (static_cast< ::PROTOBUF_NAMESPACE_ID::uint8>(tag) == (58 & 0xFF)) {
          DO_(::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::ReadMessage(
               input, mutable_data()));
        } else {
          goto handle_unusual;
        }
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0) {
          goto success;
        }
        DO_(::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SkipField(
              input, tag, _internal_metadata_.mutable_unknown_fields()));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:Meta.Meta)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:Meta.Meta)
  return false;
#undef DO_
}
#endif  // GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER

void Meta::SerializeWithCachedSizes(
    ::PROTOBUF_NAMESPACE_ID::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:Meta.Meta)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // required int32 version = 1;
  if (cached_has_bits & 0x00000004u) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32(1, this->version(), output);
  }

  // required int32 frame_id = 2;
  if (cached_has_bits & 0x00000008u) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32(2, this->frame_id(), output);
  }

  // optional .CommonProto.Image img_frame = 4;
  if (cached_has_bits & 0x00000001u) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteMessageMaybeToArray(
      4, HasBitSetters::img_frame(this), output);
  }

  // optional int32 proto_version = 5 [default = 1];
  if (cached_has_bits & 0x00000010u) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32(5, this->proto_version(), output);
  }

  // optional .MetaData.Data data = 7;
  if (cached_has_bits & 0x00000002u) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteMessageMaybeToArray(
      7, HasBitSetters::data(this), output);
  }

  if (_internal_metadata_.have_unknown_fields()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SerializeUnknownFields(
        _internal_metadata_.unknown_fields(), output);
  }
  // @@protoc_insertion_point(serialize_end:Meta.Meta)
}

::PROTOBUF_NAMESPACE_ID::uint8* Meta::InternalSerializeWithCachedSizesToArray(
    ::PROTOBUF_NAMESPACE_ID::uint8* target) const {
  // @@protoc_insertion_point(serialize_to_array_start:Meta.Meta)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // required int32 version = 1;
  if (cached_has_bits & 0x00000004u) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(1, this->version(), target);
  }

  // required int32 frame_id = 2;
  if (cached_has_bits & 0x00000008u) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(2, this->frame_id(), target);
  }

  // optional .CommonProto.Image img_frame = 4;
  if (cached_has_bits & 0x00000001u) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessageToArray(
        4, HasBitSetters::img_frame(this), target);
  }

  // optional int32 proto_version = 5 [default = 1];
  if (cached_has_bits & 0x00000010u) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(5, this->proto_version(), target);
  }

  // optional .MetaData.Data data = 7;
  if (cached_has_bits & 0x00000002u) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessageToArray(
        7, HasBitSetters::data(this), target);
  }

  if (_internal_metadata_.have_unknown_fields()) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields(), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:Meta.Meta)
  return target;
}

size_t Meta::RequiredFieldsByteSizeFallback() const {
// @@protoc_insertion_point(required_fields_byte_size_fallback_start:Meta.Meta)
  size_t total_size = 0;

  if (has_version()) {
    // required int32 version = 1;
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
        this->version());
  }

  if (has_frame_id()) {
    // required int32 frame_id = 2;
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
        this->frame_id());
  }

  return total_size;
}
size_t Meta::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:Meta.Meta)
  size_t total_size = 0;

  if (_internal_metadata_.have_unknown_fields()) {
    total_size +=
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::ComputeUnknownFieldsSize(
        _internal_metadata_.unknown_fields());
  }
  if (((_has_bits_[0] & 0x0000000c) ^ 0x0000000c) == 0) {  // All required fields are present.
    // required int32 version = 1;
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
        this->version());

    // required int32 frame_id = 2;
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
        this->frame_id());

  } else {
    total_size += RequiredFieldsByteSizeFallback();
  }
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000003u) {
    // optional .CommonProto.Image img_frame = 4;
    if (cached_has_bits & 0x00000001u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(
          *img_frame_);
    }

    // optional .MetaData.Data data = 7;
    if (cached_has_bits & 0x00000002u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(
          *data_);
    }

  }
  // optional int32 proto_version = 5 [default = 1];
  if (cached_has_bits & 0x00000010u) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
        this->proto_version());
  }

  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void Meta::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:Meta.Meta)
  GOOGLE_DCHECK_NE(&from, this);
  const Meta* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<Meta>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:Meta.Meta)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:Meta.Meta)
    MergeFrom(*source);
  }
}

void Meta::MergeFrom(const Meta& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:Meta.Meta)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = from._has_bits_[0];
  if (cached_has_bits & 0x0000001fu) {
    if (cached_has_bits & 0x00000001u) {
      mutable_img_frame()->::CommonProto::Image::MergeFrom(from.img_frame());
    }
    if (cached_has_bits & 0x00000002u) {
      mutable_data()->::MetaData::Data::MergeFrom(from.data());
    }
    if (cached_has_bits & 0x00000004u) {
      version_ = from.version_;
    }
    if (cached_has_bits & 0x00000008u) {
      frame_id_ = from.frame_id_;
    }
    if (cached_has_bits & 0x00000010u) {
      proto_version_ = from.proto_version_;
    }
    _has_bits_[0] |= cached_has_bits;
  }
}

void Meta::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:Meta.Meta)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void Meta::CopyFrom(const Meta& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:Meta.Meta)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool Meta::IsInitialized() const {
  if ((_has_bits_[0] & 0x0000000c) != 0x0000000c) return false;
  if (has_img_frame()) {
    if (!this->img_frame_->IsInitialized()) return false;
  }
  if (has_data()) {
    if (!this->data_->IsInitialized()) return false;
  }
  return true;
}

void Meta::Swap(Meta* other) {
  if (other == this) return;
  InternalSwap(other);
}
void Meta::InternalSwap(Meta* other) {
  using std::swap;
  _internal_metadata_.Swap(&other->_internal_metadata_);
  swap(_has_bits_[0], other->_has_bits_[0]);
  swap(img_frame_, other->img_frame_);
  swap(data_, other->data_);
  swap(version_, other->version_);
  swap(frame_id_, other->frame_id_);
  swap(proto_version_, other->proto_version_);
}

::PROTOBUF_NAMESPACE_ID::Metadata Meta::GetMetadata() const {
  return GetMetadataStatic();
}


// @@protoc_insertion_point(namespace_scope)
}  // namespace Meta
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::Meta::Meta* Arena::CreateMaybeMessage< ::Meta::Meta >(Arena* arena) {
  return Arena::CreateInternal< ::Meta::Meta >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>
