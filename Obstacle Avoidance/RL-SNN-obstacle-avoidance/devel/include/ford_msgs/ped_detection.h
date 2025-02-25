// Generated by gencpp from file ford_msgs/ped_detection.msg
// DO NOT EDIT!


#ifndef FORD_MSGS_MESSAGE_PED_DETECTION_H
#define FORD_MSGS_MESSAGE_PED_DETECTION_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <std_msgs/Header.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Vector3.h>
#include <std_msgs/ColorRGBA.h>

namespace ford_msgs
{
template <class ContainerAllocator>
struct ped_detection_
{
  typedef ped_detection_<ContainerAllocator> Type;

  ped_detection_()
    : header()
    , ids()
    , world_vectors()
    , left_vectors()
    , right_vectors()
    , class_strings()
    , avg_colors()
    , pincer_obs()  {
    }
  ped_detection_(const ContainerAllocator& _alloc)
    : header(_alloc)
    , ids(_alloc)
    , world_vectors(_alloc)
    , left_vectors(_alloc)
    , right_vectors(_alloc)
    , class_strings(_alloc)
    , avg_colors(_alloc)
    , pincer_obs(_alloc)  {
  (void)_alloc;
    }



   typedef  ::std_msgs::Header_<ContainerAllocator>  _header_type;
  _header_type header;

   typedef std::vector<uint32_t, typename ContainerAllocator::template rebind<uint32_t>::other >  _ids_type;
  _ids_type ids;

   typedef std::vector< ::geometry_msgs::Vector3_<ContainerAllocator> , typename ContainerAllocator::template rebind< ::geometry_msgs::Vector3_<ContainerAllocator> >::other >  _world_vectors_type;
  _world_vectors_type world_vectors;

   typedef std::vector< ::geometry_msgs::Vector3_<ContainerAllocator> , typename ContainerAllocator::template rebind< ::geometry_msgs::Vector3_<ContainerAllocator> >::other >  _left_vectors_type;
  _left_vectors_type left_vectors;

   typedef std::vector< ::geometry_msgs::Vector3_<ContainerAllocator> , typename ContainerAllocator::template rebind< ::geometry_msgs::Vector3_<ContainerAllocator> >::other >  _right_vectors_type;
  _right_vectors_type right_vectors;

   typedef std::vector<std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > , typename ContainerAllocator::template rebind<std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > >::other >  _class_strings_type;
  _class_strings_type class_strings;

   typedef std::vector< ::std_msgs::ColorRGBA_<ContainerAllocator> , typename ContainerAllocator::template rebind< ::std_msgs::ColorRGBA_<ContainerAllocator> >::other >  _avg_colors_type;
  _avg_colors_type avg_colors;

   typedef std::vector<uint8_t, typename ContainerAllocator::template rebind<uint8_t>::other >  _pincer_obs_type;
  _pincer_obs_type pincer_obs;





  typedef boost::shared_ptr< ::ford_msgs::ped_detection_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::ford_msgs::ped_detection_<ContainerAllocator> const> ConstPtr;

}; // struct ped_detection_

typedef ::ford_msgs::ped_detection_<std::allocator<void> > ped_detection;

typedef boost::shared_ptr< ::ford_msgs::ped_detection > ped_detectionPtr;
typedef boost::shared_ptr< ::ford_msgs::ped_detection const> ped_detectionConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::ford_msgs::ped_detection_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::ford_msgs::ped_detection_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::ford_msgs::ped_detection_<ContainerAllocator1> & lhs, const ::ford_msgs::ped_detection_<ContainerAllocator2> & rhs)
{
  return lhs.header == rhs.header &&
    lhs.ids == rhs.ids &&
    lhs.world_vectors == rhs.world_vectors &&
    lhs.left_vectors == rhs.left_vectors &&
    lhs.right_vectors == rhs.right_vectors &&
    lhs.class_strings == rhs.class_strings &&
    lhs.avg_colors == rhs.avg_colors &&
    lhs.pincer_obs == rhs.pincer_obs;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::ford_msgs::ped_detection_<ContainerAllocator1> & lhs, const ::ford_msgs::ped_detection_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace ford_msgs

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsFixedSize< ::ford_msgs::ped_detection_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::ford_msgs::ped_detection_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::ford_msgs::ped_detection_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::ford_msgs::ped_detection_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::ford_msgs::ped_detection_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::ford_msgs::ped_detection_<ContainerAllocator> const>
  : TrueType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::ford_msgs::ped_detection_<ContainerAllocator> >
{
  static const char* value()
  {
    return "04d576943f22d0e1e58f8bb73ed60473";
  }

  static const char* value(const ::ford_msgs::ped_detection_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x04d576943f22d0e1ULL;
  static const uint64_t static_value2 = 0xe58f8bb73ed60473ULL;
};

template<class ContainerAllocator>
struct DataType< ::ford_msgs::ped_detection_<ContainerAllocator> >
{
  static const char* value()
  {
    return "ford_msgs/ped_detection";
  }

  static const char* value(const ::ford_msgs::ped_detection_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::ford_msgs::ped_detection_<ContainerAllocator> >
{
  static const char* value()
  {
    return "Header header\n"
"uint32[] ids \n"
"geometry_msgs/Vector3[] world_vectors\n"
"geometry_msgs/Vector3[] left_vectors\n"
"geometry_msgs/Vector3[] right_vectors\n"
"string[] class_strings\n"
"std_msgs/ColorRGBA[] avg_colors\n"
"uint8[] pincer_obs\n"
"================================================================================\n"
"MSG: std_msgs/Header\n"
"# Standard metadata for higher-level stamped data types.\n"
"# This is generally used to communicate timestamped data \n"
"# in a particular coordinate frame.\n"
"# \n"
"# sequence ID: consecutively increasing ID \n"
"uint32 seq\n"
"#Two-integer timestamp that is expressed as:\n"
"# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')\n"
"# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')\n"
"# time-handling sugar is provided by the client library\n"
"time stamp\n"
"#Frame this data is associated with\n"
"string frame_id\n"
"\n"
"================================================================================\n"
"MSG: geometry_msgs/Vector3\n"
"# This represents a vector in free space. \n"
"# It is only meant to represent a direction. Therefore, it does not\n"
"# make sense to apply a translation to it (e.g., when applying a \n"
"# generic rigid transformation to a Vector3, tf2 will only apply the\n"
"# rotation). If you want your data to be translatable too, use the\n"
"# geometry_msgs/Point message instead.\n"
"\n"
"float64 x\n"
"float64 y\n"
"float64 z\n"
"================================================================================\n"
"MSG: std_msgs/ColorRGBA\n"
"float32 r\n"
"float32 g\n"
"float32 b\n"
"float32 a\n"
;
  }

  static const char* value(const ::ford_msgs::ped_detection_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::ford_msgs::ped_detection_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.header);
      stream.next(m.ids);
      stream.next(m.world_vectors);
      stream.next(m.left_vectors);
      stream.next(m.right_vectors);
      stream.next(m.class_strings);
      stream.next(m.avg_colors);
      stream.next(m.pincer_obs);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct ped_detection_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::ford_msgs::ped_detection_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::ford_msgs::ped_detection_<ContainerAllocator>& v)
  {
    s << indent << "header: ";
    s << std::endl;
    Printer< ::std_msgs::Header_<ContainerAllocator> >::stream(s, indent + "  ", v.header);
    s << indent << "ids[]" << std::endl;
    for (size_t i = 0; i < v.ids.size(); ++i)
    {
      s << indent << "  ids[" << i << "]: ";
      Printer<uint32_t>::stream(s, indent + "  ", v.ids[i]);
    }
    s << indent << "world_vectors[]" << std::endl;
    for (size_t i = 0; i < v.world_vectors.size(); ++i)
    {
      s << indent << "  world_vectors[" << i << "]: ";
      s << std::endl;
      s << indent;
      Printer< ::geometry_msgs::Vector3_<ContainerAllocator> >::stream(s, indent + "    ", v.world_vectors[i]);
    }
    s << indent << "left_vectors[]" << std::endl;
    for (size_t i = 0; i < v.left_vectors.size(); ++i)
    {
      s << indent << "  left_vectors[" << i << "]: ";
      s << std::endl;
      s << indent;
      Printer< ::geometry_msgs::Vector3_<ContainerAllocator> >::stream(s, indent + "    ", v.left_vectors[i]);
    }
    s << indent << "right_vectors[]" << std::endl;
    for (size_t i = 0; i < v.right_vectors.size(); ++i)
    {
      s << indent << "  right_vectors[" << i << "]: ";
      s << std::endl;
      s << indent;
      Printer< ::geometry_msgs::Vector3_<ContainerAllocator> >::stream(s, indent + "    ", v.right_vectors[i]);
    }
    s << indent << "class_strings[]" << std::endl;
    for (size_t i = 0; i < v.class_strings.size(); ++i)
    {
      s << indent << "  class_strings[" << i << "]: ";
      Printer<std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > >::stream(s, indent + "  ", v.class_strings[i]);
    }
    s << indent << "avg_colors[]" << std::endl;
    for (size_t i = 0; i < v.avg_colors.size(); ++i)
    {
      s << indent << "  avg_colors[" << i << "]: ";
      s << std::endl;
      s << indent;
      Printer< ::std_msgs::ColorRGBA_<ContainerAllocator> >::stream(s, indent + "    ", v.avg_colors[i]);
    }
    s << indent << "pincer_obs[]" << std::endl;
    for (size_t i = 0; i < v.pincer_obs.size(); ++i)
    {
      s << indent << "  pincer_obs[" << i << "]: ";
      Printer<uint8_t>::stream(s, indent + "  ", v.pincer_obs[i]);
    }
  }
};

} // namespace message_operations
} // namespace ros

#endif // FORD_MSGS_MESSAGE_PED_DETECTION_H
