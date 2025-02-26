// Generated by gencpp from file stage_ros/feasibleAction.msg
// DO NOT EDIT!


#ifndef STAGE_ROS_MESSAGE_FEASIBLEACTION_H
#define STAGE_ROS_MESSAGE_FEASIBLEACTION_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <std_msgs/Header.h>

namespace stage_ros
{
template <class ContainerAllocator>
struct feasibleAction_
{
  typedef feasibleAction_<ContainerAllocator> Type;

  feasibleAction_()
    : header()
    , linear_vels()
    , angular_vels()  {
    }
  feasibleAction_(const ContainerAllocator& _alloc)
    : header(_alloc)
    , linear_vels(_alloc)
    , angular_vels(_alloc)  {
  (void)_alloc;
    }



   typedef  ::std_msgs::Header_<ContainerAllocator>  _header_type;
  _header_type header;

   typedef std::vector<float, typename ContainerAllocator::template rebind<float>::other >  _linear_vels_type;
  _linear_vels_type linear_vels;

   typedef std::vector<float, typename ContainerAllocator::template rebind<float>::other >  _angular_vels_type;
  _angular_vels_type angular_vels;





  typedef boost::shared_ptr< ::stage_ros::feasibleAction_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::stage_ros::feasibleAction_<ContainerAllocator> const> ConstPtr;

}; // struct feasibleAction_

typedef ::stage_ros::feasibleAction_<std::allocator<void> > feasibleAction;

typedef boost::shared_ptr< ::stage_ros::feasibleAction > feasibleActionPtr;
typedef boost::shared_ptr< ::stage_ros::feasibleAction const> feasibleActionConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::stage_ros::feasibleAction_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::stage_ros::feasibleAction_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::stage_ros::feasibleAction_<ContainerAllocator1> & lhs, const ::stage_ros::feasibleAction_<ContainerAllocator2> & rhs)
{
  return lhs.header == rhs.header &&
    lhs.linear_vels == rhs.linear_vels &&
    lhs.angular_vels == rhs.angular_vels;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::stage_ros::feasibleAction_<ContainerAllocator1> & lhs, const ::stage_ros::feasibleAction_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace stage_ros

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsFixedSize< ::stage_ros::feasibleAction_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::stage_ros::feasibleAction_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::stage_ros::feasibleAction_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::stage_ros::feasibleAction_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::stage_ros::feasibleAction_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::stage_ros::feasibleAction_<ContainerAllocator> const>
  : TrueType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::stage_ros::feasibleAction_<ContainerAllocator> >
{
  static const char* value()
  {
    return "18cdf888fd507b687ddc0a406d22df81";
  }

  static const char* value(const ::stage_ros::feasibleAction_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x18cdf888fd507b68ULL;
  static const uint64_t static_value2 = 0x7ddc0a406d22df81ULL;
};

template<class ContainerAllocator>
struct DataType< ::stage_ros::feasibleAction_<ContainerAllocator> >
{
  static const char* value()
  {
    return "stage_ros/feasibleAction";
  }

  static const char* value(const ::stage_ros::feasibleAction_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::stage_ros::feasibleAction_<ContainerAllocator> >
{
  static const char* value()
  {
    return "Header header\n"
"float32[] linear_vels\n"
"float32[] angular_vels\n"
"\n"
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
;
  }

  static const char* value(const ::stage_ros::feasibleAction_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::stage_ros::feasibleAction_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.header);
      stream.next(m.linear_vels);
      stream.next(m.angular_vels);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct feasibleAction_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::stage_ros::feasibleAction_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::stage_ros::feasibleAction_<ContainerAllocator>& v)
  {
    s << indent << "header: ";
    s << std::endl;
    Printer< ::std_msgs::Header_<ContainerAllocator> >::stream(s, indent + "  ", v.header);
    s << indent << "linear_vels[]" << std::endl;
    for (size_t i = 0; i < v.linear_vels.size(); ++i)
    {
      s << indent << "  linear_vels[" << i << "]: ";
      Printer<float>::stream(s, indent + "  ", v.linear_vels[i]);
    }
    s << indent << "angular_vels[]" << std::endl;
    for (size_t i = 0; i < v.angular_vels.size(); ++i)
    {
      s << indent << "  angular_vels[" << i << "]: ";
      Printer<float>::stream(s, indent + "  ", v.angular_vels[i]);
    }
  }
};

} // namespace message_operations
} // namespace ros

#endif // STAGE_ROS_MESSAGE_FEASIBLEACTION_H
